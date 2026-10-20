"""Served under a sub-path: nginx at host:3000/OCR/ in front of the app.

The page works out its own base from its address (`OCR_BASE` in the template),
so a proxy that strips the prefix needs nothing here. `URL_PREFIX` is for one
that passes the path through unchanged: `/OCR/api/...` must reach `/api/...`,
and a request without the prefix must still be answered.
"""
import unittest

from werkzeug.test import Client

import app


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.client = Client(app._StripPrefix(app.app.wsgi_app, "/OCR"))

    def test_a_prefixed_request_reaches_the_route(self):
        self.assertEqual(self.client.get("/OCR/api/runs/stat").status_code, 200)
        self.assertEqual(self.client.get("/OCR/").status_code, 200)
        self.assertEqual(self.client.get("/OCR").status_code, 200)

    def test_an_unprefixed_request_still_works(self):
        self.assertEqual(self.client.get("/api/runs/stat").status_code, 200)

    def test_only_a_whole_segment_is_a_prefix(self):
        self.assertEqual(self.client.get("/OCRX/api/runs/stat").status_code, 404)

    def test_the_page_finds_its_own_base(self):
        page = app.app.test_client().get("/").get_data(as_text=True)
        self.assertIn('window.OCR_BASE_AUTO = location.pathname.replace(/\\/+$/, "")', page)
        # ...which the header's App path control may override, per browser.
        self.assertIn('localStorage.getItem("ocrAppBase")', page)
        self.assertIn('id="appBaseInput"', page)


if __name__ == "__main__":
    unittest.main()
