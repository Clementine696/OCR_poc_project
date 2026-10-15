"""The Tokens tab: in/out tokens per document, per pass, per model, from the log."""
import unittest

import runlog


def _read(case, model, detail, tokens, pages=1, sent="", status="ok", **extra):
    return {"run_type": "ocr", "case": case, "model": model, "detail": detail,
            "tokens": str(tokens), "pages": str(pages), "in_tokens": str(sent),
            "status": status, **extra}


class TokenUsageTests(unittest.TestCase):
    def test_a_new_model_gets_its_own_group(self):
        rows = [_read("sol002", "typhoon", "medium", 880, sent=4400),
                _read("sol002", "brand-new-model", "medium", 1500, sent=3000),
                _read("sol003", "brand-new-model", "medium", 1100, sent=3100)]
        out = runlog.token_usage(rows)
        models = {(m["model"], m["setting"]): m for m in out["ocr_models"]}
        self.assertEqual(models[("brand-new-model", "medium")]["documents"], 2)
        self.assertEqual(models[("brand-new-model", "medium")]["out"]["mean"], 1300)
        self.assertEqual(models[("typhoon", "medium")]["in"]["mean"], 4400)

    def test_per_page_and_older_fallbacks(self):
        rows = [_read("sol004", "typhoon", "low", 3000, pages=2,
                      server_prompt_tokens="5600")]
        entry = runlog.token_usage(rows)["ocr"][0]
        self.assertEqual(entry["in"]["mean"], 5600)
        self.assertEqual(entry["in_per_page"]["mean"], 2800)
        self.assertEqual(entry["out_per_page"]["mean"], 1500)

    def test_failed_and_retired_runs_are_left_out(self):
        rows = [_read("sol002", "typhoon", "medium", 4096, status="looped"),
                _read("sol002", "typhoon", "fast", 700)]
        self.assertEqual(runlog.token_usage(rows)["ocr"], [])

    def test_extraction_is_grouped_by_the_extracting_model_and_shape(self):
        rows = [{"run_type": "ocr", "case": "sol002", "model": "typhoon",
                 "extract_model": "gemma4:e4b", "extract_mode": "single",
                 "status": "ok", "extract_tokens": "400", "extract_in_tokens": "5200",
                 "extract_out_tokens": "470"},
                {"run_type": "extract", "case": "sol002", "model": "gemma4:e4b",
                 "extract_mode": "agentic", "status": "ok", "extract_tokens": "750"}]
        ext = {(e["model"], e["setting"]): e for e in runlog.token_usage(rows)["extract"]}
        single = ext[("gemma4:e4b", "single")]
        self.assertEqual((single["in"]["mean"], single["out"]["mean"]), (5200, 470))
        agentic = ext[("gemma4:e4b", "agentic")]
        self.assertIsNone(agentic["in"])                 # an older row: out only
        self.assertEqual(agentic["out"]["mean"], 750)

    def test_it_rides_on_totals(self):
        self.assertIn("token_usage", runlog.totals([_read("sol002", "m", "low", 900)]))


if __name__ == "__main__":
    unittest.main()
