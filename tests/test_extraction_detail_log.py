"""Download diagnostics preserve evidence without putting it in History."""
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import config
import extractionlog
import runhistory
import workspace

MARKER = "PRIVATE_RAW_REPLY_NOT_FOR_SERVER_HISTORY"
RESULT = {"mode": "agentic", "tokens": 10, "model": "mock", "documents": [
    {"document": i, "pages": [i], "doc_types": ["INVOICE"],
     "fields": {"seller_tax_id": value},
     "steps": [{"id": "seller", "keys": ["seller_tax_id"],
                "values": {"seller_tax_id": value}, "raw": [{"text": MARKER, "prompt": "question"}]}],
     "field_score": {"scalars": {"rows": [{"path": "seller_tax_id", "actual": value,
        "expected": "seller", "status": verdict, "required": True}]}}}
    for i, value, verdict in ((1, "seller", "correct"), (2, "buyer", "wrong"))]}
STATUS = {"available": True, "text_available": True, "reason": None,
          "text_reason": None, "kind": "ollama", "url": "http://mock", "model": "mock"}


class FormatTests(unittest.TestCase):
    def test_documents_values_and_replies_are_not_merged(self):
        events = list(extractionlog.events([{"status": "ok", "_extraction_result": RESULT}],
            [{"index": 3, "case": "sol014"}], {}, 2, {}))
        detail = events[1]["job"]
        self.assertEqual((detail["job"], detail["queue_position"]), (4, 2))
        self.assertEqual([(r["document"], r["pages"], r["actual"], r["status"], r["step"])
            for r in detail["field_rows"]], [(1, [1], "seller", "correct", "seller"),
                                             (2, [2], "buyer", "wrong", "seller")])
        self.assertEqual(detail["extraction"]["documents"][1]["steps"][0]["raw"][0]["text"], MARKER)
        self.assertEqual(events[-1]["field_summary"][0]["counts"], {"correct": 1, "wrong": 1})
        self.assertEqual(events[-1]["field_summary"][0]["accuracy"], .5)

    def test_missing_truth_and_cancelled_jobs_do_not_invent_expected_values(self):
        result = {"fields": {"buyer_tax_id": "A"}, "raw": MARKER, "segments": [{"pages": [1, 2]}]}
        row = extractionlog.job_detail({"status": "ok", "_extraction_result": result},
            {"index": 0, "case": "real.pdf"}, {})["field_rows"][0]
        self.assertEqual((row["status"], row["expected"], row["pages"]), ("unscored", None, [1, 2]))
        cancelled = extractionlog.job_detail({"status": "cancelled"}, {"index": 0}, {})
        self.assertIsNone(cancelled["extraction"])
        self.assertEqual(cancelled["field_rows"], [])


class StreamTests(unittest.TestCase):
    def setUp(self):
        context = workspace.use(None)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        truth = self.root / "a.md"
        truth.write_text("source transcript", encoding="utf-8")
        for target, name, kwargs in (
            (config, "LOG_DIR", {"new": self.root / "logs"}),
            (app.scoring, "cases_index", {"return_value": {"a": {"id": "a", "ground_truth": truth,
                "pdf": "a.pdf", "doc_types": ["INVOICE"]}}}),
            (app.fieldscore, "has_truth", {"return_value": True}),
            (app, "llama_status", {"return_value": STATUS}),
            (app.backends, "extract_status", {"return_value": STATUS}),
        ):
            patcher = patch.object(target, name, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = app.app.test_client()

    def run_stress(self, **overrides):
        with patch.object(app, "extract_fields", side_effect=lambda *a, **kw: copy.deepcopy(RESULT)):
            response = self.client.post("/api/stress/stream", json={"mode": "extract", "cases": ["a"],
                "jobs": 2, "warmup": 1, "concurrency": 2, "save_details": True,
                "save_log": False, "history_monitor": False, **overrides})
            self.assertEqual(response.status_code, 200)
            return [json.loads(line) for line in response.data.splitlines()]

    def test_details_download_but_never_persist_in_history_database_or_csv(self):
        events = self.run_stress()
        jobs = [e["job"] for e in events if e["event"] == "detail_log_job"]
        self.assertEqual([j["job"] for j in jobs], [2, 3])
        self.assertIn(MARKER, json.dumps(jobs))
        start = next(e for e in events if e["event"] == "detail_log_start")
        self.assertIn("source", start["provenance"]["files"]["prompts"])
        self.assertIn("sha256", start["provenance"]["files"]["prompts"])
        self.assertTrue(start["configuration"]["save_details"])
        self.assertEqual(start["plan"]["jobs"], 2)
        run_id = events[-1]["history_id"]
        self.assertNotIn(MARKER, self.client.get("/api/history/" + run_id + "?download=1").get_data(as_text=True))
        self.assertNotIn(MARKER, json.dumps(runhistory.get(run_id)))
        self.assertNotIn(MARKER.encode(), (self.root / "logs/run_history.sqlite3").read_bytes())
        self.assertFalse((self.root / "logs/runs.csv").exists())

    def test_disabled_option_emits_no_diagnostics(self):
        self.assertFalse(any(e["event"].startswith("detail_log")
                             for e in self.run_stress(save_details=False)))

    def test_api_validates_boolean_and_ignores_option_for_non_extraction_modes(self):
        response = self.client.post("/api/stress/stream", json={"save_details": "yes"})
        self.assertEqual(response.status_code, 400)
        for mode in ("ocr", "classify"):
            self.assertFalse(app._stress_config({"mode": mode, "save_details": True})["save_details"])

    def test_export_delay_is_outside_benchmark_wall_clock(self):
        original = extractionlog.events
        def slow(*args):
            time.sleep(.1)
            yield from original(*args)
        with patch.object(extractionlog, "events", side_effect=slow):
            began = time.perf_counter()
            events = self.run_stress(jobs=1, warmup=0)
            elapsed = time.perf_counter() - began
        self.assertGreater(elapsed - events[-1]["report"]["wall_seconds"], .08)


if __name__ == "__main__":
    unittest.main()
