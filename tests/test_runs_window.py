"""Runs averaged is the newest N rows that match the filters, cut before compiling.

2026-10-21, at the user's request: *100 latest run thats match all condition*.
The card used to compile every row and trim per group afterwards, so a small
window was no faster than the whole log.
"""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import runlog


class RunsWindowTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        root = Path(self._dir.name)
        path = root / "runs.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=runlog.COLUMNS)
            writer.writeheader()
            # Oldest first, as the log is written: a, b, a, b, a, b, a.
            for i in range(7):
                writer.writerow({"timestamp": f"2026-10-21T10:00:0{i}",
                                 "file": f"doc{i}.pdf", "case": "sol001",
                                 "model": "a" if i % 2 == 0 else "b",
                                 "status": "ok", "char_accuracy": "90"})
        self._patch = patch.multiple(runlog, LOG_DIR=root, LOG_PATH=path)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._dir.cleanup()

    def test_window_keeps_only_the_newest_rows(self):
        totals = app._runs_payload(50, window=3)["totals"]
        self.assertEqual(totals["in_window"], 3)
        self.assertEqual(totals["runs"], 3)
        self.assertEqual(totals["matched"], 7)
        self.assertEqual(totals["logged"], 7)

    def test_window_counts_rows_after_the_filter(self):
        payload = app._runs_payload(50, window=2, include={"model": ["a"]})
        totals = payload["totals"]
        self.assertEqual(totals["matched"], 4)
        self.assertEqual(totals["in_window"], 2)
        self.assertEqual(totals["runs"], 2)
        # The raw table and the pickers are still the whole log.
        self.assertEqual(len(payload["runs"]), 7)

    def test_zero_is_the_whole_log(self):
        self.assertEqual(app._runs_payload(50, window=0)["totals"]["in_window"], 7)


if __name__ == "__main__":
    unittest.main()
