"""Each browser tab its own pickers and switches, and its own sweep.

The app is deployed for several people at once. Every control on the page used
to be process state, so one person's choice moved everyone's next run; now a
request naming a workspace (`X-OCR-Workspace`, or `?ws=` on the page load) has
its own copy, and a request naming none is the default workspace -- the module
globals the CLI and these tests have always used.
"""
import threading
import unittest
from unittest import mock

import app
import backends
import settings
import workspace

A = {workspace.HEADER: "tab-aaaaaaaa"}
B = {workspace.HEADER: "tab-bbbbbbbb"}

OLLAMA = "http://ollama:11434"
PROBE = {"kind": "ollama", "reachable": True, "model": None,
         "models": [{"name": "typhoon-ocr", "vision": True},
                    {"name": "gemma4:e4b", "vision": True},
                    {"name": "qwen2.5vl:3b", "vision": True}],
         "vision": None, "slots": None, "reason": None, "url": OLLAMA}


class WorkspaceTestCase(unittest.TestCase):
    def setUp(self):
        workspace.reset()
        self.addCleanup(workspace.reset)
        self.client = app.app.test_client()
        self.saved = (app._extract_mode, app._loop_guard, app._read_floor,
                      app._ocr_profile, backends._num_ctx,
                      (settings.OCR_TEMPERATURE, settings.OCR_TOP_P, settings.OCR_TOP_K),
                      (app.TABLE_FIX_AGENT, app.TABLE_IDENTIFY_AGENT, app.TABLE_CONCAT_AGENT),
                      backends._active, dict(backends._chosen), list(backends._endpoints))
        self.addCleanup(self.restore)

    def restore(self):
        (app._extract_mode, app._loop_guard, app._read_floor, app._ocr_profile,
         backends._num_ctx, sampling, agents, backends._active, chosen,
         backends._endpoints) = self.saved
        settings.OCR_TEMPERATURE, settings.OCR_TOP_P, settings.OCR_TOP_K = sampling
        app.TABLE_FIX_AGENT, app.TABLE_IDENTIFY_AGENT, app.TABLE_CONCAT_AGENT = agents
        backends._chosen.clear()
        backends._chosen.update(chosen)


class SwitchTests(WorkspaceTestCase):
    def test_one_tabs_switches_are_not_anothers(self):
        before = (app.extract_mode(), app.loop_guard(), app.read_floor(),
                  backends.num_ctx(), app.table_agents()["fix"])
        mode = "single" if before[0] == "agentic" else "agentic"
        c = self.client
        self.assertEqual(c.post("/api/extract/mode", json={"mode": mode}, headers=A).status_code, 200)
        c.post("/api/ocr/loop-guard", json={"loop_guard": not before[1]}, headers=A)
        c.post("/api/ocr/read-floor", json={"min_read_pct": 12}, headers=A)
        c.post("/api/table-agents", json={"fix": not before[4]}, headers=A)
        c.post("/api/ocr/sampling", json={"temperature": 0.7}, headers=A)
        with mock.patch.object(backends, "status", return_value={"url": OLLAMA}):
            c.post("/api/context", json={"num_ctx": 16384}, headers=A)

        with workspace.use(workspace.find(A[workspace.HEADER])):
            self.assertEqual(app.extract_mode(), mode)
            self.assertEqual(app.loop_guard(), not before[1])
            self.assertAlmostEqual(app.read_floor(), 0.12)
            self.assertEqual(app.table_agents()["fix"], not before[4])
            self.assertEqual(settings.ocr_sampling()["temperature"], 0.7)
            self.assertEqual(backends.num_ctx(), 16384)

        # B never asked for anything, and starts where the process did.
        self.assertEqual(c.get("/api/extract/mode", headers=B).get_json()["mode"], before[0])
        self.assertEqual(c.get("/api/ocr/loop-guard", headers=B).get_json()["loop_guard"], before[1])
        # And the default workspace -- the CLI's -- is untouched.
        self.assertEqual((app.extract_mode(), app.loop_guard(), app.read_floor(),
                          backends.num_ctx(), app.table_agents()["fix"]), before)
        self.assertEqual(settings.ocr_sampling()["temperature"], self.saved[5][0])

    def test_a_new_tab_starts_from_startup_not_from_the_cli(self):
        workspace.seal()
        start = app.extract_mode()
        app.set_extract_mode("single" if start == "agentic" else "agentic")
        self.assertEqual(self.client.get("/api/extract/mode", headers=B)
                         .get_json()["mode"], start)


class ServerTests(WorkspaceTestCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(backends, "probe", return_value=PROBE)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.stopped = []
        stop = mock.patch.object(backends, "stop_model",
                                 side_effect=lambda url, m: self.stopped.append(m) or True)
        stop.start()
        self.addCleanup(stop.stop)
        resident = mock.patch.object(
            backends, "loaded_models",
            side_effect=lambda url: (["typhoon-ocr", "gemma4:e4b", "qwen2.5vl:3b"]
                                     if url == OLLAMA else []))
        resident.start()
        self.addCleanup(resident.stop)
        # Every tab starts on the one Ollama, as a deployment with one would.
        backends._active = OLLAMA

    def _ask(self, headers):
        with workspace.use(workspace.find(headers[workspace.HEADER])):
            return backends.active_url(), backends.status()["model"]

    def test_picking_a_model_moves_only_that_tab(self):
        c = self.client
        c.post("/api/servers", json={"url": OLLAMA, "model": "typhoon-ocr"}, headers=A)
        c.post("/api/servers", json={"url": OLLAMA, "model": "gemma4:e4b"}, headers=B)
        self.assertEqual(self._ask(A), (OLLAMA, "typhoon-ocr"))
        self.assertEqual(self._ask(B), (OLLAMA, "gemma4:e4b"))
        self.assertNotIn(OLLAMA, backends._chosen)  # the default never picked

    def test_a_switch_does_not_unload_another_tabs_model(self):
        c = self.client
        c.post("/api/servers", json={"url": OLLAMA, "model": "typhoon-ocr"}, headers=A)
        self.stopped.clear()
        c.post("/api/servers", json={"url": OLLAMA, "model": "gemma4:e4b"}, headers=B)
        # B's switch frees what nobody is using, and keeps A's reader resident.
        self.assertIn("qwen2.5vl:3b", self.stopped)
        self.assertNotIn("typhoon-ocr", self.stopped)
        self.assertNotIn("gemma4:e4b", self.stopped)


class SweepTests(WorkspaceTestCase):
    def test_a_sweep_in_one_tab_does_not_block_another(self):
        started, finish = threading.Event(), threading.Event()
        ws_a = workspace.enter(A[workspace.HEADER])

        def sweep():
            with workspace.use(ws_a):
                app.begin_sweep()
                started.set()
                finish.wait(5)
                app.end_sweep()

        thread = threading.Thread(target=sweep)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(finish.set)
        started.wait(5)
        with workspace.use(ws_a):
            self.assertTrue(app.sweeps_running())
        with workspace.use(workspace.enter(B[workspace.HEADER])):
            self.assertFalse(app.sweeps_running())
            self.assertFalse(app.cancel_sweep())
        with workspace.use(None):
            self.assertFalse(app.sweeps_running())
        with workspace.use(ws_a):
            self.assertTrue(app.cancel_sweep())

    def test_a_busy_workspace_is_never_evicted(self):
        ws = workspace.enter(A[workspace.HEADER])
        workspace.hold()
        ws.seen = 0
        workspace._last_sweep = 0
        workspace.enter(B[workspace.HEADER])
        self.assertIs(workspace.find(A[workspace.HEADER]), ws)


class ThreadTests(WorkspaceTestCase):
    def test_bind_carries_the_workspace_into_a_thread(self):
        ws = workspace.enter(A[workspace.HEADER])
        seen = []
        thread = threading.Thread(target=workspace.bind(lambda: seen.append(workspace.current())))
        thread.start()
        thread.join()
        self.assertIs(seen[0], ws)


class JobTests(WorkspaceTestCase):
    def test_jobs_are_capped_per_workspace(self):
        from PIL import Image
        page = [Image.new("RGB", (4, 4))]
        workspace.enter(A[workspace.HEADER])
        first_a = app.register_job(page)
        workspace.enter(B[workspace.HEADER])
        for _ in range(app.MAX_JOBS + 1):
            app.register_job(page)
        self.assertIn(first_a, app._jobs)


if __name__ == "__main__":
    unittest.main()
