"""Runs averaged is each model's newest N matching rows, cut before compiling.

2026-10-21, at the user's request: *100 latest run thats match all condition*,
then *make it 100 per model*. The card used to compile every row and trim per
group afterwards, so a small window was no faster than the whole log.
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


class RunsWindowTests(unittest.TestCase):
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

    def tearDown(self):
        self._patch.stop()
        self._dir.cleanup()

    def test_window_keeps_each_models_newest_rows(self):
        totals = app._runs_payload(50, window=2)["totals"]
        # Two of `a` and two of `b`, not the log's newest two (one of each).
        self.assertEqual(totals["in_window"], 4)
        self.assertEqual(totals["runs"], 4)
        self.assertEqual(totals["matched"], 7)
        self.assertEqual(totals["logged"], 7)

    def test_a_model_with_fewer_runs_keeps_all_of_them(self):
        # `a` has 4 and `b` has 3: a window of 4 takes every row.
        self.assertEqual(app._runs_payload(50, window=4)["totals"]["in_window"], 7)

    def test_window_counts_rows_after_the_filter(self):
        payload = app._runs_payload(50, window=2, include={"model": ["a"]})
        totals = payload["totals"]
        self.assertEqual(totals["matched"], 4)
        self.assertEqual(totals["in_window"], 2)
        self.assertEqual(totals["runs"], 2)
        # The raw table and the pickers are still the whole log.
        self.assertEqual(len(payload["runs"]), 7)
        self.assertEqual({f["value"] for f in payload["facets"]["model"]}, {"a", "b"})

    def test_zero_is_the_whole_log(self):
        self.assertEqual(app._runs_payload(50, window=0)["totals"]["in_window"], 7)

    def test_cached_parse_follows_a_new_row(self):
        self.assertEqual(len(runlog.read(limit=10 ** 6)), 7)
        self.assertEqual(runlog.log_facets()["model"][0]["runs"], 4)
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=runlog.COLUMNS).writerow(_row(7, "c"))
        self.assertEqual(len(runlog.read(limit=10 ** 6)), 8)
        self.assertIn("c", {f["value"] for f in runlog.log_facets()["model"]})

    def test_cached_rows_are_not_shared_between_callers(self):
        runlog.read(limit=10 ** 6)[0]["model"] = "edited"
        self.assertNotEqual(runlog.read(limit=10 ** 6)[0]["model"], "edited")

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
