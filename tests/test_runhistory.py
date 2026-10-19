"""History persists exact snapshots and terminal states independently of the CSV."""

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import app
import config
import runhistory


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.patch = patch.object(config, "LOG_DIR", Path(self.directory.name))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.client = app.app.test_client()

    def stream(self, events, configuration=None):
        return app._history_stream((json.dumps(e) + "\n" for e in events),
                                   "stress", configuration or {"concurrency": 4, "jobs": 20})

    def test_round_trip_and_list_does_not_include_large_report(self):
        configuration = {"concurrency": 4, "jobs": 20, "label": "ทดสอบ"}
        run_id = runhistory.start("stress", configuration)
        configuration["jobs"] = 999
        plan = {"mode": "full", "seed": 42}
        result = {"report": {"latency": {"p99": 1.234}, "missing": None}}
        runhistory.save_plan(run_id, plan)
        runhistory.finish(run_id, "completed", result)
        item = runhistory.get(run_id)
        self.assertEqual(item["configuration"]["jobs"], 20)
        self.assertEqual(item["configuration"]["label"], "ทดสอบ")
        self.assertEqual(item["plan"], plan)
        self.assertEqual(item["result"], result)
        self.assertIsNotNone(item["finished_at"])
        self.assertNotIn("result", runhistory.list_runs()["runs"][0])

    def test_pagination_and_concurrent_writes_keep_every_run(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(lambda i: runhistory.start("stress", {"jobs": i}), range(8)))
        first, second = runhistory.list_runs(4, 0), runhistory.list_runs(4, 4)
        self.assertEqual(first["total"], 8)
        self.assertEqual({r["id"] for r in first["runs"] + second["runs"]}, set(ids))

    def test_unfinished_runs_from_other_instances_are_not_declared_failed(self):
        pending = runhistory.start("stress", {})
        done = runhistory.start("stress", {})
        runhistory.finish(done, "completed", {"report": {}})
        with patch.object(runhistory, "PROCESS_ID", "new-process"):
            self.assertEqual(runhistory.get(pending)["status"], "unfinished")
            self.assertEqual(runhistory.get(done)["status"], "completed")

    def test_completion_saved_before_client_receives_done_and_not_overwritten(self):
        stream = self.stream([{"event": "plan", "seed": 42},
                              {"event": "done", "report": {"error_rate": 0}}])
        plan = json.loads(next(stream))
        done = json.loads(next(stream))
        self.assertEqual(plan["history_id"], done["history_id"])
        self.assertEqual(runhistory.get(done["history_id"])["status"], "completed")
        stream.close()
        runhistory.finish(done["history_id"], "error", error="late failure")
        self.assertEqual(runhistory.get(done["history_id"])["status"], "completed")

    def test_stopped_and_error_reports_remain_available(self):
        for stopped, rate, expected in [(True, 0, "stopped"),
                                         (False, 0.5, "completed_with_errors")]:
            events = [json.loads(line) for line in self.stream([
                {"event": "plan"}, {"event": "done", "stopped": stopped,
                                     "report": {"error_rate": rate, "errors": ["failure"]}}])]
            item = runhistory.get(events[-1]["history_id"])
            self.assertEqual(item["status"], expected)
            self.assertEqual(item["result"]["report"]["errors"], ["failure"])

    def test_disconnect_and_missing_completion_are_not_success(self):
        stream = self.stream([{"event": "plan"}, {"event": "progress", "done": 2}])
        run_id = json.loads(next(stream))["history_id"]
        stream.close()
        self.assertEqual(runhistory.get(run_id)["status"], "interrupted")
        events = [json.loads(line) for line in self.stream([{"event": "plan"}])]
        self.assertEqual(runhistory.get(events[0]["history_id"])["status"], "interrupted")

    def test_failed_stream_keeps_progress_and_error(self):
        def broken():
            yield json.dumps({"event": "progress", "done": 3})
            raise ValueError("engine unavailable")
        stream = app._history_stream(broken(), "stress", {})
        run_id = json.loads(next(stream))["history_id"]
        with self.assertRaises(ValueError):
            next(stream)
        item = runhistory.get(run_id)
        self.assertEqual(item["status"], "error")
        self.assertEqual(item["progress"]["done"], 3)
        self.assertIn("engine unavailable", item["error"])

    def test_storage_failure_warns_without_discarding_report(self):
        with patch.object(runhistory, "start", side_effect=OSError("read-only disk")):
            events = [json.loads(line) for line in self.stream([
                {"event": "done", "report": {"completed": 5}}])]
        self.assertEqual(events[-1]["report"]["completed"], 5)
        self.assertIn("read-only disk", events[-1]["history_error"])

    def test_saved_monitor_snapshots_include_final_reading_and_gpu(self):
        cfg = {"history_monitor": True, "metrics_every": 0, "gpu_url": "http://gpu",
               "reader": {"kind": "vllm", "url": "http://vllm"}}
        sample = {"at": 123, "snapshot": {"kv_free": 0.5, "running": 4},
                  "gpu": {"free_mb": 3584}, "rates": {"generation_tps": 100}}
        with patch.object(app, "_monitor_scrape", return_value=(sample, 200)) as scrape:
            events = [json.loads(line) for line in self.stream([
                {"event": "plan"}, {"event": "progress"},
                {"event": "done", "report": {}}], cfg)]
        item = runhistory.get(events[-1]["history_id"])
        self.assertEqual(len(item["result"]["monitor_samples"]), 3)
        self.assertEqual(item["result"]["monitor_samples"][-1], {**sample, "role": "reader"})
        scrape.assert_called_with({**cfg["reader"], "role": "reader"}, "http://vllm", "http://gpu")

    def test_monitor_never_probes_non_vllm_or_disabled_sampling(self):
        for kind, enabled in [("ollama", True), ("vllm", False)]:
            with patch.object(app, "_monitor_scrape") as scrape:
                list(self.stream([{"event": "done", "report": {}}], {
                    "history_monitor": enabled, "reader": {"kind": kind, "url": "http://x"}}))
            scrape.assert_not_called()

    def test_monitor_retains_latest_300_with_total_sample_count(self):
        cfg = {"history_monitor": True, "metrics_every": 0,
               "reader": {"kind": "vllm", "url": "http://vllm"}}
        samples = iter(({"at": i, "snapshot": {}}, 200) for i in range(306))
        source = [{"event": "plan"}] + [{"event": "progress"}] * 304 + [
            {"event": "done", "report": {}}]
        with patch.object(app, "_monitor_scrape", side_effect=lambda *_: next(samples)):
            events = [json.loads(line) for line in self.stream(source, cfg)]
        result = runhistory.get(events[-1]["history_id"])["result"]
        self.assertEqual(len(result["monitor_samples"]), 300)
        self.assertEqual(result["monitor_samples"][0]["at"], 6)
        self.assertEqual(result["monitor_samples"][-1]["at"], 305)
        self.assertEqual(result["monitor_samples_total"], 306)

    def test_both_monitor_servers_and_live_alerts_are_retained(self):
        cfg = {"history_monitor": True, "metrics_every": 5,
               "reader": {"kind": "vllm", "url": "http://reader"},
               "extractor": {"kind": "vllm", "url": "http://extractor"},
               "gpu_url": "http://reader-gpu", "gpu_extract_url": "http://extractor-gpu"}
        alert = {"event": "alert", "job": 2, "case": "sol002", "message": "looped"}
        with patch.object(app, "_monitor_scrape", side_effect=lambda info, url, gpu: (
                {"url": url, "gpu_url": gpu, "snapshot": {}, "at": 1}, 200)):
            events = [json.loads(line) for line in self.stream([
                {"event": "plan"}, alert, {"event": "done", "report": {}}], cfg)]
        result = runhistory.get(events[-1]["history_id"])["result"]
        self.assertEqual(result["alerts"], [alert])
        self.assertEqual({s["url"] for s in result["monitor_samples"]},
                         {"http://reader", "http://extractor"})
        self.assertEqual(result["monitor_samples"][-1]["gpu_url"], "http://extractor-gpu")

    def test_routes_return_exact_saved_data_and_json_download(self):
        run_id = runhistory.start("stress", {"concurrency": 8, "jobs": 50})
        runhistory.finish(run_id, "completed", {"report": {"completed": 50}})
        listed = self.client.get('/api/history?limit=1').get_json()
        self.assertEqual(listed["runs"][0]["id"], run_id)
        result = self.client.get('/api/history/' + run_id + '?download=1')
        self.assertEqual(result.get_json()["result"]["report"]["completed"], 50)
        self.assertIn('attachment;', result.headers["Content-Disposition"])
        self.assertEqual(self.client.get('/api/history/invalid').status_code, 404)
        self.assertEqual(self.client.get('/api/history?limit=oops').status_code, 400)

    def test_stress_endpoint_persists_effective_settings_without_csv_rows(self):
        info = {"kind": "ollama", "url": "http://extract", "model": "model",
                "text_available": True, "text_reason": ""}
        source = lambda *_: (json.dumps(e) + '\n' for e in [
            {"event": "plan", "seed": 42}, {"event": "done", "report": {"error_rate": 0}}])
        with patch.object(app.backends, "extract_status", return_value=info), \
             patch.object(app, "_stress_run", side_effect=source), \
             patch.object(app.runlog, "record", side_effect=AssertionError("CSV changed")):
            result = self.client.post('/api/stress/stream', json={
                "mode": "extract", "concurrency": 3, "jobs": 5, "warmup": 2})
            events = [json.loads(line) for line in result.data.splitlines()]
        item = runhistory.get(events[-1]["history_id"])
        self.assertEqual(item["configuration"]["concurrency"], 3)
        self.assertEqual(item["configuration"]["jobs"], 5)
        self.assertEqual(item["configuration"]["warmup"], 2)
        self.assertEqual(item["configuration"]["extractor"]["url"], "http://extract")


if __name__ == '__main__':
    unittest.main()
