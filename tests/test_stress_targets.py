"""A full-pipeline stress run samples /metrics on BOTH servers it uses."""
import unittest
from unittest import mock

import app

READ = {"kind": "vllm", "url": "http://read:8000", "model": "typhoon"}
EXTRACT = {"kind": "vllm", "url": "http://extract:8000", "model": "qwen"}
OLLAMA = {"kind": "ollama", "url": "http://local:11434", "model": "gemma"}


class TargetTests(unittest.TestCase):
    def test_two_vllm_servers_are_two_targets(self):
        self.assertEqual(app._stress_targets(READ, EXTRACT), [
            {"role": "reader", "url": "http://read:8000"},
            {"role": "extractor", "url": "http://extract:8000"}])

    def test_one_server_for_both_passes_is_one_target(self):
        self.assertEqual(app._stress_targets(READ, dict(EXTRACT, url=READ["url"])),
                         [{"role": "reader + extractor", "url": "http://read:8000"}])

    def test_a_server_that_is_not_vllm_is_never_asked(self):
        self.assertEqual(app._stress_targets(OLLAMA, EXTRACT),
                         [{"role": "extractor", "url": "http://extract:8000"}])
        self.assertEqual(app._stress_targets(READ, None),
                         [{"role": "reader", "url": "http://read:8000"}])
        self.assertEqual(app._stress_targets(OLLAMA, None), [])

    def test_a_sample_reads_every_target_and_keeps_going_past_a_failure(self):
        asked = []

        def fake(url):
            asked.append(url)
            if "extract" in url:
                raise OSError("down")
            return {"vllm:num_requests_running": 2.0}

        targets = app._stress_targets(READ, EXTRACT)
        with mock.patch.object(app, "_vllm_metrics", side_effect=fake):
            views, snaps, errors = app._stress_sample(targets, {}, 1.0)
        self.assertEqual(asked, ["http://read:8000", "http://extract:8000"])
        self.assertEqual([v["role"] for v in views], ["reader"])
        self.assertIn("http://read:8000", snaps)
        self.assertEqual(len(errors), 1)
        self.assertIn("extractor", errors[0])


if __name__ == "__main__":
    unittest.main()
