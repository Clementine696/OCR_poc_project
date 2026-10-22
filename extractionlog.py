"""Download-only extraction diagnostics. No filesystem writes or model requests."""
import hashlib
from collections import Counter
from pathlib import Path


def provenance(prompt_file, app_file, settings):
    """Capture the code/prompt identity once, before the timed run."""
    files = {}
    for label, path in (("prompts", prompt_file), ("app", app_file)):
        try:
            data = Path(path).read_bytes()
            files[label] = {"sha256": hashlib.sha256(data).hexdigest()}
            if label == "prompts":
                files[label]["source"] = data.decode("utf-8")
        except (OSError, UnicodeError) as err:
            files[label] = {"error": str(err)}
    return {"files": files, "settings": settings}


def job_detail(record, spec, source, warmup=0):
    """Keep the original extraction and flatten scores without merging documents."""
    result = record.get("_extraction_result")
    rows = []
    if isinstance(result, dict):
        segments = result.get("segments") or []
        for number, document in enumerate(result.get("documents") or [result], 1):
            fields = document.get("fields") or {}
            steps = document.get("steps") or []
            step_keys = {key: step["id"] for step in steps for key in step.get("keys") or []}
            pages = document.get("pages") or (segments[number - 1].get("pages")
                                             if number <= len(segments) else []) or []
            score = document.get("field_score") or {}
            judged = []
            for section in ("scalars", "line_items", "income_items", "other_fields"):
                judged.extend((score.get(section) or {}).get("rows") or [])
            checked = {row["path"] for row in judged}
            # Missing ground truth is unscored, never an empty expected value.
            judged += [{"path": key, "actual": value, "expected": None, "status": "unscored"}
                       for key, value in fields.items() if key not in checked
                       and not isinstance(value, (list, dict))]
            for row in judged:
                path = row["path"]
                key = path.split("[", 1)[0]
                rows.append({**row, "document": document.get("document", number),
                             "pages": pages, "doc_types": document.get("doc_types") or [],
                             "step": step_keys.get(key), "actual_returned": key in fields})
    return {"job": spec["index"] + 1, "queue_position": spec["index"] - warmup + 1,
            "case": spec.get("case"), "source": source,
            "status": record.get("status"), "error": record.get("error", ""),
            "timing": {key: record[key] for key in ("wait_s", "total_s", "ocr_s", "extract_s")
                       if key in record},
            "field_rows": rows, "extraction": result}


def field_summary(counts):
    """Pooled verdict counts, with invented values reported separately."""
    rows = []
    for field, verdicts in sorted(counts.items()):
        scored = sum(verdicts.get(key, 0) for key in ("correct", "partial", "wrong", "missed"))
        rows.append({"field": field, "counts": dict(verdicts), "scored": scored,
                     "accuracy": ((verdicts.get("correct", 0) + .5 * verdicts.get("partial", 0))
                                  / scored if scored else None)})
    return rows


def events(records, specs, sources, warmup, identity):
    """Serialize only after the runner stopped its benchmark clock; never save on disk."""
    yield {"event": "detail_log_start", "format_version": 1, "provenance": identity}
    counts = {}
    for record, spec in zip(records, specs):
        detail = job_detail(record, spec, sources.get(spec.get("case"), {}), warmup)
        for row in detail["field_rows"]:
            counts.setdefault(row["path"], Counter())[row["status"]] += 1
        yield {"event": "detail_log_job", "job": detail}
    yield {"event": "detail_log_end", "jobs": len(records), "field_summary": field_summary(counts)}
