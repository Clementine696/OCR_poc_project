"""The stress test's "bypass the server's prompt cache" switch."""
import unittest
from unittest.mock import patch

import app
import backends


def _info(kind):
    return {"kind": kind, "model": "m", "url": "http://s"}


class BypassTests(unittest.TestCase):
    def tearDown(self):
        while backends.prefix_cache_bypassed():
            backends.restore_prefix_cache()

    def test_nothing_is_added_while_off(self):
        for kind in ("vllm", "llama.cpp", "ollama", "openai"):
            extras = backends.request_extras(_info(kind))
            self.assertNotIn("cache_salt", extras, kind)
            self.assertNotIn("cache_prompt", extras, kind)

    def test_vllm_gets_a_fresh_salt_per_request(self):
        backends.bypass_prefix_cache()
        a = backends.request_extras(_info("vllm"))["cache_salt"]
        b = backends.request_extras(_info("vllm"))["cache_salt"]
        self.assertNotEqual(a, b)
        self.assertGreaterEqual(len(a), 40)

    def test_llama_cpp_turns_cache_prompt_off(self):
        backends.bypass_prefix_cache()
        with patch.object(backends, "num_ctx", return_value=0):
            self.assertIs(backends.request_extras(_info("llama.cpp"))["cache_prompt"], False)

    def test_ollama_and_generic_openai_are_unchanged(self):
        backends.bypass_prefix_cache()
        for kind in ("ollama", "openai"):
            extras = backends.request_extras(_info(kind))
            self.assertNotIn("cache_salt", extras, kind)
            self.assertNotIn("cache_prompt", extras, kind)
            self.assertIn("not possible", backends.cache_bypass_effect(kind))

    def test_the_pass_2_body_carries_it(self):
        backends.bypass_prefix_cache()
        with patch.object(backends, "num_ctx", return_value=8192):
            _, body = backends.structured_request(
                [{"role": "user", "content": "x"}], None, 64, _info("vllm"))
        self.assertIn("cache_salt", body)

    def test_it_is_a_count(self):
        backends.bypass_prefix_cache()
        backends.bypass_prefix_cache()
        backends.restore_prefix_cache()
        self.assertTrue(backends.prefix_cache_bypassed())
        backends.restore_prefix_cache()
        self.assertFalse(backends.prefix_cache_bypassed())
        backends.restore_prefix_cache()        # never goes negative
        self.assertFalse(backends.prefix_cache_bypassed())

    def test_stress_config_reads_the_flag(self):
        with patch.object(app, "_stress_cases", return_value=["sol002"]), \
             patch.object(app.backends, "extract_status",
                          return_value={"text_available": True, "text_reason": ""}):
            self.assertTrue(app._stress_config({"mode": "extract", "no_cache": True})["no_cache"])
            self.assertFalse(app._stress_config({"mode": "extract"})["no_cache"])


if __name__ == "__main__":
    unittest.main()
