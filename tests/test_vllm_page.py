"""The page as vLLM is sent it: enlarged when small, cap lifted, max_tokens kept in the window."""
import unittest
from unittest.mock import patch

from PIL import Image

import app
import settings


def _vllm(window=8192):
    models = [{"name": "typhoon", "max_model_len": window}] if window else []
    return {"kind": "vllm", "model": "typhoon", "models": models}


class VllmPageTests(unittest.TestCase):
    def test_small_page_is_enlarged_to_the_floor(self):
        with patch.object(settings, "VLLM_MIN_PIXELS", 2_000_000):
            out = app.vllm_page(Image.new("RGB", (800, 1100), "white"))
        self.assertAlmostEqual(out.width * out.height, 2_000_000, delta=5000)
        self.assertAlmostEqual(out.width / out.height, 800 / 1100, places=2)

    def test_large_page_is_untouched(self):
        page = Image.new("RGB", (1500, 2000), "white")
        with patch.object(settings, "VLLM_MIN_PIXELS", 2_000_000):
            self.assertIs(app.vllm_page(page), page)

    def test_zero_floor_disables(self):
        page = Image.new("RGB", (400, 500), "white")
        with patch.object(settings, "VLLM_MIN_PIXELS", 0):
            self.assertIs(app.vllm_page(page), page)

    def test_fields_lift_the_cap_and_clamp_max_tokens(self):
        page = Image.new("RGB", (1758, 2275), "white")      # ~4 MP
        with patch.object(settings, "VLLM_MAX_PIXELS", 4_194_304):
            fields = app.vllm_page_fields(page, _vllm(8192))
        self.assertEqual(fields["mm_processor_kwargs"], {"max_pixels": 4_194_304})
        image_tokens = -(-1758 * 2275 // app.VLLM_PIXELS_PER_TOKEN)
        self.assertEqual(fields["max_tokens"],
                         min(app.MAX_NEW_TOKENS, 8192 - image_tokens - app.VLLM_TEXT_MARGIN))
        self.assertLess(fields["max_tokens"], app.MAX_NEW_TOKENS)

    def test_clamp_counts_the_capped_pixels_only(self):
        page = Image.new("RGB", (3000, 4000), "white")      # 12 MP, capped to 2 MP
        with patch.object(settings, "VLLM_MAX_PIXELS", 2_000_000):
            fields = app.vllm_page_fields(page, _vllm(8192))
        self.assertEqual(fields["max_tokens"],
                         min(app.MAX_NEW_TOKENS, 8192 - 1954 - app.VLLM_TEXT_MARGIN))

    def test_no_window_no_clamp(self):
        fields = app.vllm_page_fields(Image.new("RGB", (100, 100)), _vllm(None))
        self.assertNotIn("max_tokens", fields)

    def test_tiny_window_keeps_a_floor(self):
        fields = app.vllm_page_fields(Image.new("RGB", (2000, 2000)), _vllm(2048))
        self.assertEqual(fields["max_tokens"], 256)

    def test_zero_cap_sends_no_override(self):
        with patch.object(settings, "VLLM_MAX_PIXELS", 0):
            fields = app.vllm_page_fields(Image.new("RGB", (100, 100)), _vllm(None))
        self.assertEqual(fields, {})


if __name__ == "__main__":
    unittest.main()
