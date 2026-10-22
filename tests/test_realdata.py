"""realData/: real files with no solution, listed, served and stress-tested."""
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import workspace


STATUS = {"available": True, "text_available": True, "reason": None,
          "text_reason": None, "kind": "ollama", "url": "http://mock",
          "model": "mock-model"}


class RealDataTests(unittest.TestCase):
    def setUp(self):
        context = workspace.use(None)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "realData"
        (self.root / "batch1").mkdir(parents=True)
        (self.root / "a.pdf").write_bytes(b"%PDF-1.4 a")
        (self.root / "b.JPEG").write_bytes(b"jpeg b")
        (self.root / "batch1" / "c.png").write_bytes(b"png c")
        (self.root / "notes.txt").write_text("not a document", encoding="utf-8")
        for target, name, kwargs in (
            (app, "REAL_DIR", {"new": self.root}),
            (app, "llama_status", {"return_value": STATUS}),
            (app.backends, "extract_status", {"return_value": STATUS}),
        ):
            patcher = patch.object(target, name, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = app.app.test_client()

    def cfg(self, **kwargs):
        return app._stress_config({"mode": "ocr", "source": "realdata", **kwargs})

    def test_listing_walks_subfolders_and_skips_unreadable_types(self):
        data = self.client.get("/api/realdata").get_json()
        self.assertEqual([f["name"] for f in data["files"]],
                         ["a.pdf", "b.JPEG", "batch1/c.png"])
        self.assertEqual(data["types"], ["jpg", "pdf", "png"])

    def test_file_is_served_and_traversal_refused(self):
        res = self.client.get("/api/realdata/file?name=batch1/c.png")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.data, b"png c")
        for bad in ("../x.pdf", "notes.txt", "missing.pdf", ""):
            with self.subTest(name=bad):
                self.assertEqual(
                    self.client.get("/api/realdata/file?name=" + bad).status_code, 404)

    def test_random_draws_from_every_file_of_the_chosen_types(self):
        self.assertEqual(self.cfg()["cases"], ["a.pdf", "b.JPEG", "batch1/c.png"])
        self.assertEqual(self.cfg(types=["jpeg", ".png"])["cases"],
                         ["b.JPEG", "batch1/c.png"])
        with self.assertRaises(ValueError):
            self.cfg(types=["tiff"])

    def test_chosen_files_must_be_in_the_folder_and_of_a_chosen_type(self):
        self.assertEqual(self.cfg(cases=["a.pdf"])["cases"], ["a.pdf"])
        with self.assertRaises(ValueError):
            self.cfg(cases=["a.pdf"], types=["png"])
        with self.assertRaises(ValueError):
            self.cfg(cases=["sol001"])

    def test_only_modes_that_start_from_a_read(self):
        self.assertEqual(self.cfg(mode="full")["source"], "realdata")
        for mode in ("extract", "classify"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.cfg(mode=mode)

    def test_empty_folder_is_refused(self):
        with patch.object(app, "REAL_DIR", self.root / "batch1" / "none"):
            with self.assertRaises(ValueError):
                self.cfg()

    def test_run_reads_and_extracts_without_scoring(self):
        cfg = self.cfg(mode="full", cases=["a.pdf"], jobs=2, warmup=1, seed=2,
                       history_monitor=False)
        with patch.object(app, "_stress_pages", return_value=([], {})) as pages, \
                patch.object(app, "_stress_read", return_value=(["a transcript"], False, False)), \
                patch.object(app, "extract_fields", return_value={"tokens": 1}) as extract, \
                patch.object(app.scoring, "evaluate") as evaluate, \
                patch.object(app.runlog, "case_counts") as counts:
            events = [json.loads(line) for line in app._stress_run(cfg, app.MultiCancel())]
        pages.assert_called_once_with("a.pdf", cfg["detail"], "realdata")
        evaluate.assert_not_called()
        counts.assert_not_called()
        self.assertEqual({call.args[2] for call in extract.call_args_list}, {None})
        self.assertEqual(events[0]["source"], "realdata")
        report = events[-1]["report"]
        self.assertEqual(report["completed"], 2)
        self.assertIsNone(report["ocr"]["char_accuracy"])
        self.assertIsNone(report["extract"]["field_accuracy"])

    def test_pages_are_named_as_realdata(self):
        page = types.SimpleNamespace()
        with patch.object(app, "load_pages", return_value=([page], None)), \
                patch.object(app, "fit_pixels", side_effect=lambda p, b: p), \
                patch.object(app, "trim_margins", side_effect=lambda p: p), \
                patch.object(app, "image_data_uri", return_value="data:"):
            pages, source = app._stress_pages("batch1/c.png", "medium", "realdata")
        self.assertEqual(source["origin"], "realdata")
        self.assertEqual(source["name"], "batch1/c.png")
        self.assertEqual(len(pages), 1)


if __name__ == "__main__":
    unittest.main()
