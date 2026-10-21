"""The run-log card's caches change how fast it answers, never what it says.

2026-10-21, at the user's request: *keep the same logic but i want to save time
and data transfer of log retriving*. The parsed CSV, the pickers and the whole
payload are kept until the file moves, and JSON replies are gzipped.
"""
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import runlog


def _row(i, model):
    return {"timestamp": f"2026-10-21T10:00:{i:02d}", "file": f"doc{i}.pdf",
            "case": "sol001", "model": model, "status": "ok", "char_accuracy": "90"}


class RunsCacheTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        root = Path(self._dir.name)
        self.path = root / "runs.csv"
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=runlog.COLUMNS)
            writer.writeheader()
            # Oldest first, as the log is written: a, b, a, b, a, b, a.
            for i in range(7):
                writer.writerow(_row(i, "a" if i % 2 == 0 else "b"))
        self._patch = patch.multiple(runlog, LOG_DIR=root, LOG_PATH=self.path)
        self._patch.start()
        app._runs_cache.clear()

    def tearDown(self):
        self._patch.stop()
        app._runs_cache.clear()
        self._dir.cleanup()

    def _append(self, i, model):
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=runlog.COLUMNS).writerow(_row(i, model))

    def test_window_is_still_per_document(self):
        # Every row is one document, so a window of 2 is its newest 2 rows.
        totals = app._runs_payload(50, window=2)["totals"]
        self.assertEqual(totals["runs"], 2)
        self.assertEqual(totals["logged"], 7)
        self.assertEqual(totals["window"], 2)

    def test_cached_payload_equals_a_fresh_compile(self):
        cached = app._runs_payload(50, window=2, include={"model": ["a"]})
        again = app._runs_payload(50, window=2, include={"model": ["a"]})
        fresh = app._compile_runs(50, 2, None, {"model": ["a"]}, None, False, False)
        for payload in (cached, again):
            self.assertEqual(json.loads(json.dumps(payload, default=str)),
                             json.loads(json.dumps(fresh, default=str)))

    def test_a_new_row_moves_every_cache(self):
        self.assertEqual(app._runs_payload(50, window=0)["totals"]["logged"], 7)
        self.assertEqual(runlog.log_facets()["model"][0]["runs"], 4)
        self._append(7, "c")
        payload = app._runs_payload(50, window=0)
        self.assertEqual(payload["totals"]["logged"], 8)
        self.assertEqual(len(payload["runs"]), 8)
        self.assertIn("c", {f["value"] for f in payload["facets"]["model"]})

    def test_cached_rows_are_not_shared_between_callers(self):
        runlog.read(limit=10 ** 6)[0]["model"] = "edited"
        self.assertNotEqual(runlog.read(limit=10 ** 6)[0]["model"], "edited")

    def test_a_reply_does_not_edit_the_cached_payload(self):
        first = app._runs_payload(50)
        first["totals"]["environment"] = "edited"
        self.assertIsInstance(app._runs_payload(50)["totals"]["environment"], dict)

    def test_the_payload_is_gzipped_for_a_client_that_accepts_it(self):
        client = app.app.test_client()
        plain = client.post("/api/runs/query", json={"window": 2})
        packed = client.post("/api/runs/query", json={"window": 2},
                             headers={"Accept-Encoding": "gzip, deflate"})
        self.assertNotIn("Content-Encoding", plain.headers)
        self.assertEqual(packed.headers["Content-Encoding"], "gzip")
        self.assertEqual(json.loads(gzip.decompress(packed.data)), plain.get_json())


if __name__ == "__main__":
    unittest.main()
