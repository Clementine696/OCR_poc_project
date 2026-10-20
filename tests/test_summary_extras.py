"""Extra fields on/off are different extraction measurements."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import runlog
import settings


def row(extras="1", mode="single", seconds=100, correct=9, **kwargs):
    return {"timestamp": "2026-10-20T12:00:00", "file": "invoice.pdf", "case": "sol002",
            "run_type": "extract", "model": "extractor", "backend": "vllm",
            "extract_mode": mode, "extract_extras": extras, "status": "ok",
            "extract_seconds": seconds, "p1_present": 10, "p1_correct": correct,
            "p1_scored": 10, "p1_partial": 0, "other_fields": 0, **kwargs}


class SummaryExtrasTests(unittest.TestCase):
    def presentation(self, rows, **kwargs):
        with runlog.view(window=0):
            return runlog.presentation(rows, bias=False, **kwargs)

    def test_shapes_split_accuracy_timing_and_errors_for_both_settings(self):
        rows = [row("1", seconds=100, correct=9), row("0", seconds=10, correct=8),
                row("0", seconds=999, status="error"),
                row("1", mode="agentic", seconds=60), row("0", mode="agentic", seconds=6)]
        result = self.presentation(rows)
        shapes = {(r["mode"], r["extras"]): r for r in result["extract_shapes"]}
        self.assertEqual(len(shapes), 4)
        on, off = shapes[("single", True)], shapes[("single", False)]
        self.assertEqual((on["seconds"]["mean"], off["seconds"]["mean"]), (100, 10))
        self.assertEqual((on["value"]["mean"], off["value"]["mean"]), (90, 80))
        self.assertEqual((on["failure_rate"], off["failure_rate"]), (0, 50))
        self.assertEqual((on["runs"], off["runs"]), (1, 2))
        self.assertEqual({r["extras"] for r in result["extract_models"]}, {True, False})

    def test_legacy_runs_join_on_even_with_zero_or_missing_extra_count(self):
        legacy = row()
        legacy.pop("extract_extras")
        unknown_count = {**legacy, "other_fields": ""}
        result = self.presentation([legacy, unknown_count, row("1"), row("0")])
        groups = {r["extras"]: r for r in result["extract_shapes"]}
        self.assertEqual(groups[True]["runs"], 3)
        self.assertEqual(groups[False]["runs"], 1)
        self.assertNotIn("extract_extras", legacy)  # compilation must not rewrite history

    def test_recent_window_is_independent_for_on_and_off(self):
        with runlog.view(window=1):
            groups = {r["extras"]: r for r in runlog.presentation(
                [row("1", seconds=100), row("0", seconds=10),
                 row("1", seconds=200), row("0", seconds=20)], bias=False)["extract_shapes"]}
        self.assertEqual((groups[True]["seconds"]["mean"], groups[False]["seconds"]["mean"]),
                         (100, 10))
        self.assertEqual((groups[True]["runs"], groups[False]["runs"]), (1, 1))

    def test_slow_extras_run_is_not_an_outlier_of_the_faster_off_runs(self):
        rows = [row("0", seconds=1) for _ in range(10)] + [row("1", seconds=100)]
        with runlog.view(window=0):
            result = runlog.presentation(rows, bias=True, single_source=False)
        on = next(r for r in result["extract_shapes"] if r["extras"])
        self.assertEqual(on["seconds"]["mean"], 100)
        self.assertFalse(any(step["rule"] == "time_outlier" for step in result["bias"]))

    def test_weak_on_setting_does_not_hide_good_off_setting(self):
        rows = [row("1", status="error") for _ in range(5)] + [row("0") for _ in range(5)]
        with patch.object(settings, "PRESENT_MIN_RUNS", 3), runlog.view(window=0):
            result = runlog.presentation(rows, bias=True, single_source=False)
        self.assertEqual([r["extras"] for r in result["extract_shapes"]], [False])
        self.assertEqual(result["weak"]["extract"][0]["model"], "extractor")
        self.assertTrue(result["weak"]["extract"][0]["extras"])

    def test_document_means_remain_balanced_within_each_setting(self):
        result = self.presentation([row("1", seconds=100) for _ in range(10)]
                                   + [row("1", case="sol003", seconds=20), row("0", seconds=5)])
        on = next(r for r in result["extract_shapes"] if r["extras"])
        self.assertEqual(on["seconds"]["mean"], 60)

    def test_ocr_groups_do_not_split_by_extraction_switch(self):
        rows = [row(extras, run_type="ocr", seconds=5, char_accuracy=95, detail="medium")
                for extras in ("1", "0")]
        self.assertEqual(len(self.presentation(rows)["ocr_models"]), 1)


class ExtrasLogTests(unittest.TestCase):
    def test_logging_uses_the_result_switch_for_success_failure_and_multiple_documents(self):
        for extras in (True, False):
            for result in ({"fields": {}}, {"error": "failed"},
                           {"documents": [{"fields": {}}, {"fields": {}}]}):
                with self.subTest(extras=extras, result=result):
                    cells = runlog._extract_cells({"extracted": {**result, "extras": extras}})
                    self.assertEqual(cells["extract_extras"], int(extras))
        self.assertEqual(runlog._extract_cells({})["extract_extras"], "")
        self.assertIn("extract_extras", runlog.EXTRACT_COLUMNS)

    def test_old_csv_migrates_without_changing_existing_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "runs.csv"
            columns = [c for c in runlog.COLUMNS if c != "extract_extras"]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerow({"file": "old.pdf", "extract_seconds": "123"})
            with patch.multiple(runlog, LOG_DIR=root, LOG_PATH=path):
                saved = runlog.record({"extracted": {"extras": False, "mode": "single",
                                                     "fields": {}, "seconds": 10}},
                                      {"name": "new.pdf"})
                self.assertEqual(saved["extract_extras"], 0)
            with path.open(newline="", encoding="utf-8-sig") as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(records[0]["extract_seconds"], "123")
            self.assertEqual(records[0]["extract_extras"], "")
            self.assertEqual(records[1]["extract_extras"], "0")


if __name__ == "__main__":
    unittest.main()
