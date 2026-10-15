"""Pages/s and token figures on a random-test round and over the run."""
import unittest

import randomtest


def full_round(pages=2, read_in=1000, read_out=400, ex_in=600, ex_out=200,
               read_s=8.0, ex_s=2.0):
    return {"page_count": pages, "seconds": read_s,
            "in_tokens": read_in, "out_tokens": read_out,
            "extracted": {"seconds": ex_s,
                          "server_time": {"in_tokens": ex_in, "out_tokens": ex_out}}}


class RoundTests(unittest.TestCase):
    def test_full_round_adds_both_passes(self):
        out = randomtest.summarise_round(full_round())
        self.assertEqual(out["pages"], 2)
        self.assertEqual(out["pages_per_second"], 0.2)
        self.assertEqual((out["in_tokens"], out["out_tokens"], out["tokens"]),
                         (1600, 600, 2200))
        self.assertEqual(out["tokens_per_page"], 1100.0)

    def test_ocr_round_has_no_pass_two(self):
        payload = full_round()
        del payload["extracted"]
        out = randomtest.summarise_round(payload)
        self.assertEqual(out["tokens"], 1400)
        self.assertEqual(out["pages_per_second"], 0.25)

    def test_fields_round_has_tokens_and_no_pages(self):
        out = randomtest.summarise_round(
            {"extracted": {"seconds": 3.0,
                           "server_time": {"in_tokens": 500, "out_tokens": 100}}})
        self.assertEqual(out["tokens"], 600)
        self.assertEqual(out["pages"], 0)
        self.assertIsNone(out["pages_per_second"])
        self.assertIsNone(out["tokens_per_page"])

    def test_unknown_part_blanks_the_total(self):
        out = randomtest.summarise_round(full_round(read_in=None))
        self.assertIsNone(out["in_tokens"])
        self.assertIsNone(out["tokens"])
        self.assertIsNone(out["tokens_per_page"])
        self.assertEqual(out["out_tokens"], 600)


class RunTotalTests(unittest.TestCase):
    def test_totals(self):
        rounds = [randomtest.summarise_round(full_round()),
                  randomtest.summarise_round(full_round(pages=1, read_s=4.0, ex_s=1.0)),
                  randomtest.summarise_round(full_round(read_in=None)),
                  randomtest.summarise_round(
                      {"extracted": {"seconds": 5.0, "server_time":
                                     {"in_tokens": 300, "out_tokens": 100}}})]
        totals = randomtest.run_totals(rounds)
        self.assertEqual(totals["pages"], 5)
        self.assertEqual(totals["pages_per_second"], round(5 / 30.0, 3))
        self.assertEqual(totals["tokens"], 2200 + 2200 + 400)
        self.assertEqual(totals["token_rounds"], 3)
        self.assertEqual(totals["rounds"], 4)
        # The fields round's tokens are in the total, not in tokens/page.
        self.assertEqual(totals["tokens_per_page"], round(4400 / 3, 1))

    def test_empty_run(self):
        totals = randomtest.run_totals([])
        self.assertEqual(totals["pages"], 0)
        self.assertIsNone(totals["pages_per_second"])
        self.assertIsNone(totals["tokens_per_page"])

    def test_throughput_line(self):
        line = randomtest._throughput(randomtest.summarise_round(full_round()))
        self.assertEqual(line, "pages 2  0.20 pages/s  tokens 2,200 "
                               "(in 1,600 / out 600)  1,100 tokens/page")


if __name__ == "__main__":
    unittest.main()
