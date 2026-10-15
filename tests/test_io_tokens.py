"""In/out tokens: counted per request whether or not the server timed it."""
import unittest
from unittest import mock

import runlog
import servertime
import stress

VLLM = {"kind": "vllm", "url": "http://gpu:8000", "reachable": True, "model": "m"}


class IoTokenTests(unittest.TestCase):
    def test_usage_is_counted_without_a_server_time(self):
        with mock.patch.object(servertime.settings, "SERVER_TIMING_METRICS", False):
            clock = servertime.Clock(VLLM)
            clock.usage({"prompt_tokens": 900, "completion_tokens": 40})
            rec = clock.finish()
        self.assertIsNone(rec["server_seconds"])
        self.assertEqual((rec["in_tokens"], rec["out_tokens"]), (900, 40))

    def test_the_servers_own_count_wins(self):
        rec = {"server_prompt_tokens": 1000, "server_generated_tokens": 50}
        self.assertEqual(servertime.io_tokens(rec, {"prompt_tokens": 1, "generated_tokens": 2}),
                         (1000, 50))
        # llama.cpp without cache_n: only the computed part is known.
        self.assertEqual(servertime.io_tokens({"server_prefill_tokens": 300}, {}), (300, None))

    def test_the_tally_sums_every_request_or_says_nothing(self):
        with mock.patch.object(servertime.settings, "SERVER_TIMING_METRICS", False):
            previous = servertime.open_tally()
            for usage in ({"prompt_tokens": 100, "completion_tokens": 10},
                          {"prompt_tokens": 200, "completion_tokens": 20}):
                clock = servertime.Clock(VLLM)
                clock.usage(usage)
                clock.finish()
            done = servertime.close_tally(previous)
        self.assertEqual((done["in_tokens"], done["out_tokens"]), (300, 30))

        with mock.patch.object(servertime.settings, "SERVER_TIMING_METRICS", False):
            previous = servertime.open_tally()
            clock = servertime.Clock(VLLM)
            clock.usage({"prompt_tokens": 100, "completion_tokens": 10})
            clock.finish()
            servertime.Clock(VLLM).finish()          # a request with no usage
            done = servertime.close_tally(previous)
        self.assertIsNone(done["in_tokens"])


class StressReportTests(unittest.TestCase):
    def test_per_page_and_per_file_min_max(self):
        records = [
            {"status": "ok", "pages": 2, "page_in": [1000, 1200], "page_out": [300, 500],
             "extract_s": 1.0, "extract_in": 4000, "extract_out": 600},
            {"status": "ok", "pages": 1, "page_in": [900], "page_out": [None],
             "extract_s": 1.0, "extract_in": 5000, "extract_out": 800},
        ]
        rep = stress.report(records, 10.0, "full", 2)
        page = rep["ocr"]["tokens_per_page"]
        self.assertEqual((page["in"]["min"], page["in"]["max"], page["in"]["n"]), (900, 1200, 3))
        self.assertEqual((page["out"]["min"], page["out"]["max"], page["out"]["n"]), (300, 500, 2))
        files = rep["extract_tokens_per_file"]
        self.assertEqual((files["in"]["min"], files["in"]["max"]), (4000, 5000))
        self.assertEqual(files["out"]["mean"], 700.0)

    def test_classify_counts_as_extraction(self):
        rep = stress.report([{"status": "ok", "pages": 100, "extract_in": 9000,
                              "extract_out": 120}], 5.0, "classify", 1)
        self.assertEqual(rep["extract_tokens_per_file"]["in"]["max"], 9000)


class RunLogTests(unittest.TestCase):
    def test_both_passes_write_their_tokens(self):
        cells = runlog._server_time_cells({"in_tokens": 7000, "out_tokens": 900})
        self.assertEqual((cells["extract_in_tokens"], cells["extract_out_tokens"]), (7000, 900))
        self.assertEqual(runlog._server_time_cells(None)["extract_in_tokens"], "")
        self.assertIn("extract_in_tokens", runlog.EXTRACT_COLUMNS)
        for name in ("in_tokens", "out_tokens", "extract_in_tokens", "extract_out_tokens"):
            self.assertIn(name, runlog.COLUMNS)


if __name__ == "__main__":
    unittest.main()
