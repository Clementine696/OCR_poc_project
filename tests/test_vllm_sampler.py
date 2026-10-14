"""vLLM is sent the penalties Ollama actually applies to the same model."""
import unittest
from unittest.mock import patch

import backends
import settings


def _vllm(model):
    return {"kind": "vllm", "model": model, "url": "http://v"}


class VllmSamplerTests(unittest.TestCase):
    def test_qwen_gets_the_modelfile_presence_penalty(self):
        extras = backends.request_extras(_vllm("Qwen/Qwen3.5-9B"))
        self.assertEqual(extras["presence_penalty"], 1.5)
        self.assertEqual(extras["repetition_penalty"], 1.1)

    def test_typhoon_gets_no_global_penalty(self):
        # Measured 2026-10-14: vLLM's unwindowed penalty does not stop the table loop
        # and costs ~12 points on sol003. DRY in vllm_xargs stops it instead.
        extras = backends.request_extras(_vllm("scb10x/typhoon-ocr1.5-3b"))
        self.assertEqual(extras["repetition_penalty"], 1.0)
        self.assertNotIn("presence_penalty", extras)

    def test_unlisted_model_gets_ollamas_default(self):
        extras = backends.request_extras(_vllm("google/gemma-4-e4b-it"))
        self.assertEqual(extras["repetition_penalty"], 1.1)
        self.assertNotIn("presence_penalty", extras)

    def test_other_backends_are_untouched(self):
        for kind in ("ollama", "llama.cpp", "openai"):
            extras = backends.request_extras({"kind": kind, "model": "qwen3.5:9b"})
            self.assertNotIn("presence_penalty", extras, kind)
            self.assertNotIn("repetition_penalty", extras, kind)

    def test_it_replaces_the_shared_penalty_in_the_sent_body(self):
        with patch.object(settings, "REPETITION_PENALTY", 1.0), \
             patch.object(backends, "num_ctx", return_value=8192):
            url, body = backends.structured_request(
                [{"role": "user", "content": "x"}], None, 64, _vllm("qwen3.5-9b"))
        self.assertTrue(url.endswith("/v1/chat/completions"))
        self.assertEqual(body["repetition_penalty"], 1.1)
        self.assertEqual(body["presence_penalty"], 1.5)
        self.assertEqual(body["temperature"], 0)

    def test_vllm_gets_dry_in_vllm_xargs(self):
        extras = backends.request_extras(_vllm("typhoon-ocr-1-5"))
        self.assertEqual(extras["vllm_xargs"], {
            "dry_multiplier": settings.DRY_MULTIPLIER, "dry_base": 1.75,
            "dry_allowed_length": settings.VLLM_DRY_ALLOWED_LENGTH,
            "dry_penalty_last_n": settings.DRY_PENALTY_LAST_N or -1,
            "repeat_penalty": settings.VLLM_WINDOWED_PENALTY})
        for kind in ("ollama", "llama.cpp", "openai"):
            extras = backends.request_extras({"kind": kind, "model": "typhoon"})
            self.assertNotIn("vllm_xargs", extras, kind)

    def test_dry_off_switches(self):
        # DRY goes, the windowed penalty is still sent so the server's default cannot apply.
        for name, value in (("VLLM_DRY", False), ("DRY_MULTIPLIER", 0)):
            with patch.object(settings, name, value):
                xargs = backends.request_extras(_vllm("typhoon"))["vllm_xargs"]
            self.assertEqual(xargs, {"repeat_penalty": settings.VLLM_WINDOWED_PENALTY}, name)

    def test_off_switch_and_overrides(self):
        with patch.object(settings, "VLLM_MATCH_OLLAMA", False):
            self.assertEqual(settings.vllm_sampler("qwen"), {})
        with patch.object(settings, "VLLM_PRESENCE_PENALTY", 0.5), \
             patch.object(settings, "VLLM_FREQUENCY_PENALTY", 0.2):
            fields = settings.vllm_sampler("gemma")
        self.assertEqual(fields["presence_penalty"], 0.5)
        self.assertEqual(fields["frequency_penalty"], 0.2)
        self.assertEqual(fields["repetition_penalty"], 1.1)


if __name__ == "__main__":
    unittest.main()
