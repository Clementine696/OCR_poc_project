"""Benchmark endpoints stay on their configured ports across model restarts."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


class ModelDefaultsTests(unittest.TestCase):
    def settings(self, **overrides):
        env = dict(os.environ)
        for key in ('OCR_ENDPOINTS', 'LLAMA_URL', 'EXTRACT_URL', 'AUTO_SELECT_SERVER'):
            env.pop(key, None)
        env.update(overrides)
        result = subprocess.run([
            sys.executable, '-c',
            'import json, backends, settings; print(json.dumps({'
            '"ocr": backends.active_url(), "extract": backends.extract_url(), '
            '"endpoints": backends.endpoints(), "auto": settings.AUTO_SELECT_SERVER}))'
        ], cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True,
            text=True, check=True)
        return json.loads(result.stdout)

    def test_default_ports_are_separate_and_no_fallback_is_selected(self):
        result = self.settings()
        self.assertEqual(result['ocr'], 'http://127.0.0.1:8000')
        self.assertEqual(result['extract'], 'http://127.0.0.1:8001')
        self.assertEqual(result['endpoints'], [result['ocr'], result['extract']])
        self.assertFalse(result['auto'])

    def test_explicit_deployment_settings_take_precedence(self):
        result = self.settings(LLAMA_URL='http://reader:18000', EXTRACT_URL='http://extractor:18001',
                               AUTO_SELECT_SERVER='1')
        self.assertEqual(result['ocr'], 'http://reader:18000')
        self.assertEqual(result['extract'], 'http://extractor:18001')
        self.assertTrue(result['auto'])

    def test_explicit_empty_extraction_url_preserves_shared_server_setup(self):
        result = self.settings(OCR_ENDPOINTS='http://one:8080,http://two:11434', EXTRACT_URL='')
        self.assertEqual(result['ocr'], 'http://one:8080')
        self.assertEqual(result['extract'], result['ocr'])
        self.assertEqual(result['endpoints'], ['http://one:8080', 'http://two:11434'])
