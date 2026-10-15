# Tokens per document, per process, per model

In = prompt tokens sent (image tokens included for OCR). Out = tokens generated.
OCR figures are per whole document (all pages), from `logs/runs.csv`, finished runs only.
Extraction in/out (gemma4:e4b) were measured 2026-10-15, truth-fed, on Ollama, one run each;
they count every pass-2 request of the file, classify/segment/table agents included.
Extraction out for qwen3.5:9b is the log's mean (the log never recorded extraction input).

| doc | type | pages | OCR in, typhoon 3b medium | OCR out, typhoon 3b | OCR out, typhoon 2b | OCR out, dots.mocr | extract in/out, gemma4:e4b single | extract in/out, gemma4:e4b agentic | extract out, qwen3.5:9b single | extract out, qwen3.5:9b agentic |
|---|---|---|---|---|---|---|---|---|---|---|
| sol001 | SoA+invoice | 1 | — | 2,142 | 2,245 | 2,867 | 9,626 / 658 | 23,868 / 1,208 | 440 | 916 |
| sol002 | invoice | 1 | 4,415 | 883 | 861 | 1,806 | 5,274 / 466 | 13,396 / 756 | 486 | 700 |
| sol003 | receipt+tax inv | 1 | 4,366 | 795 | 884 | 1,929 | 5,378 / 442 | 12,046 / 540 | 317 | 323 |
| sol004 | receipt+tax inv | 2 | 8,728 | 3,090 | 3,189 | 4,716 | 13,056 / 609 | 31,772 / 1,187 | 418 | 314 |
| sol005 | receipt+tax inv | 1 | 4,362 | 1,333 | 1,298 | 2,060 | 7,157 / 470 | 16,933 / 811 | 595 | 546 |
| sol006 | credit note+tax inv | 1 | 4,362 | 1,110 | 1,251 | 2,159 | 7,024 / 588 | 15,974 / 454 | 346 | 639 |
| sol007 | receipt+tax inv | 1 | — | 1,278 | 1,224 | 2,223 | 7,383 / 1,417 | 16,638 / 879 | 511 | 457 |
| sol008 | credit note | 1 | 4,390 | 559 | 602 | 1,101 | 4,373 / 519 | 10,377 / 324 | 217 | 438 |
| sol009 | credit note | 1 | — | 822 | 792 | 1,910 | 5,801 / 320 | 12,781 / 353 | 398 | 313 |
| sol010 | credit note | 1 | — | 1,008 | 1,027 | 1,643 | 4,909 / 377 | 12,959 / 543 | 314 | 597 |
| sol011 | receipt+tax inv | 1 | 4,404 | 905 | 957 | 2,216 | 6,522 / 348 | 14,942 / 759 | 233 | 694 |
| sol012 | receipt+tax inv | 2 | 8,726 | 2,258 | 2,204 | 3,952 | 13,790 / 1,127 | 45,467 / 910 | — | 989 |
| sol013 | tax inv | 1 | — | 1,181 | 1,082 | 2,123 | 8,405 / 671 | 32,496 / 885 | 565 | 891 |
| sol014 | credit note | 2 | 8,724 | 1,736 | 1,745 | 3,484 | 11,578 / 1,064 | 38,967 / 1,553 | 1,088 | 1,134 |
| sol015 | billing note | 7 | 30,534 | 7,240 | 7,125 | 11,758 | 55,772 / 3,213 | 189,062 / 5,640 | 6,456 | 4,509 |
| sol016 | WHT cert | 1 | 4,362 | 1,885 | 1,877 | 3,366 | 3,684 / 504 | 15,450 / 880 | — | 640 |
| sol017 | WHT cert | 1 | 1,714 | 2,108 | 2,330 | 3,742 | 4,101 / 762 | 19,203 / 1,324 | 503 | 492 |
| sol018 | WHT cert | 1 | 1,714 | 2,526 | 2,088 | — | 4,276 / 548 | 20,778 / 1,236 | 507 | 647 |
| sol019 | WHT cert | 1 | 1,398 | 2,011 | 2,151 | — | 4,054 / 423 | 18,780 / 1,182 | 682 | 702 |
| sol020 | WHT cert | 1 | 2,348 | 2,448 | 2,127 | 3,712 | 4,287 / 604 | 20,877 / 1,367 | 672 | 543 |
| sol021 | tax inv | 4 | — | 5,118 | 5,428 | — | 29,672 / 1,797 | 97,924 / 2,860 | 2,191 | 2,626 |
| sol022 | tax inv | 4 | 17,401 | 4,011 | 4,623 | — | 28,849 / 2,300 | 91,893 / 2,758 | 1,195 | 1,267 |
| sol023 | credit note+tax inv | 13 | 40,713 | 25,073 | 26,292 | — | 84,504 / 2,544 | 223,785 / 3,665 | 2,706 | 2,544 |
