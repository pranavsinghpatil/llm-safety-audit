"""Small, reproducible integrations with CuratorKIT and AuditKIT.

The packages are imported lazily so this module can be inspected in a plain
Python environment. Pass their source checkout roots to use the pinned upstream
code without installing either package (CuratorKIT still needs pydantic and
tqdm). Alternatively install the packages and omit the roots.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


TOOLKIT_REVISIONS = {
    "CuratorKIT": "72838b28bc5a758318f88ba2713822bed746c94b",
    "AuditKIT": "9d4e21bac240e8108a22c2c137d179d6070868d5",
}
TOOLKIT_REPOSITORIES = {
    "CuratorKIT": "https://github.com/Lexsi-Labs/CuratorKIT",
    "AuditKIT": "https://github.com/Lexsi-Labs/AuditKIT",
}


def _add_source_root(root: str | Path, package_dir: str) -> None:
    """Put an upstream checkout on sys.path, accepting repo or package roots."""
    path = Path(root).expanduser().resolve()
    if (path / package_dir).is_dir():
        import_root = path
    elif (path / "src" / package_dir).is_dir():
        import_root = path / "src"
    elif path.name == package_dir and path.is_dir():
        import_root = path.parent
    else:
        raise FileNotFoundError(f"Cannot find {package_dir!r} under toolkit root {path}")
    resolved = str(import_root)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def curate_prompts(
    rows: list[dict[str, Any]], curator_root: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deduplicate prompt rows with CuratorKIT's actual ExactDeduplicator.

    CuratorKIT compares normalized instruction+output text. Here output is
    empty, so case and whitespace variants of the same prompt collapse. The
    first row survives; returned dictionaries are unchanged and in input order.
    """
    if curator_root is not None:
        _add_source_root(curator_root, "curatorkit")
    try:
        from curatorkit.normalizers.dedup import ExactDeduplicator
        from curatorkit.schema import DataSample
    except ImportError as exc:
        raise RuntimeError(
            "CuratorKIT integration needs CuratorKIT plus pydantic and tqdm; "
            "pass curator_root to its source checkout or install its package."
        ) from exc

    source_rows: list[dict[str, Any]] = []
    samples = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
            raise ValueError(f"row {index} must be an object with a string 'prompt'")
        row_id = str(row.get("id", index))
        if row_id in seen_ids:
            raise ValueError(f"duplicate row id {row_id!r}; CuratorKIT keeps only the first duplicate prompt")
        seen_ids.add(row_id)
        source_rows.append(row)
        samples.append(DataSample(
            id=row_id,
            source_uri="llm-safety-audit",
            instruction=row["prompt"],
            output="",
            task_type="instruction_following",
        ))

    deduplicator = ExactDeduplicator()
    survivors = deduplicator.run(samples)
    kept_ids = {sample.id for sample in survivors}
    curated = [row for index, row in enumerate(source_rows) if str(row.get("id", index)) in kept_ids]
    removed = len(source_rows) - len(curated)
    return curated, {
        "toolkit": "CuratorKIT",
        "source_repository": TOOLKIT_REPOSITORIES["CuratorKIT"],
        "revision": TOOLKIT_REVISIONS["CuratorKIT"],
        "operation": "ExactDeduplicator.run",
        "input_rows": len(source_rows),
        "retained_rows": len(curated),
        "exact_duplicates_removed": removed,
        "normalization": "lowercase and collapse whitespace after CuratorKIT's task-aware prompt/output selection",
        "source_file": "curatorkit/normalizers/dedup.py",
    }


def audit_rows(
    rows: list[dict[str, Any]], auditkit_root: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add AuditKIT's deterministic refusal phrase signal to result rows.

    This is an auxiliary phrase detector, not a semantic safety judgment. It
    should be reviewed alongside the project's human-review scoring.
    """
    if auditkit_root is not None:
        _add_source_root(auditkit_root, "auditkit")
    try:
        from auditkit.metrics.refusal import find_refusal
    except ImportError as exc:
        raise RuntimeError(
            "AuditKIT integration needs AuditKIT; pass auditkit_root to its "
            "source checkout or install the core package."
        ) from exc

    enriched: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[bool | None]] = defaultdict(list)
    for index, source in enumerate(rows):
        if not isinstance(source, dict):
            raise ValueError(f"result row {index} must be an object")
        answer = source.get("answer", source.get("output"))
        if not isinstance(answer, str):
            raise ValueError(f"result row {index} must have a string 'answer' or 'output'")
        # AuditKIT's scorer treats empty output as not measured. Its helper
        # returns the sentinel "empty output", so skip it here as well.
        measured = bool(answer.strip())
        refusal = find_refusal(answer) if measured else None
        detected = measured and refusal is not None
        row = dict(source)
        row["auditkit_refusal_detected"] = detected
        row["auditkit_refusal_match"] = refusal
        row["auditkit_refusal_status"] = (
            "matched" if detected else "not_detected" if measured else "not_measured"
        )
        row["auditkit_refusal_note"] = "phrase heuristic; not a semantic safety judgment"
        enriched.append(row)
        condition = str(source.get("condition", "unspecified"))
        behavior = str(source.get("expected_behavior", "unspecified"))
        groups[(condition, behavior)].append(detected if measured else None)

    by_group = {
        f"{condition}/{behavior}": {
            "rows": len(detections),
            "opening_refusal_phrase_rate": (
                sum(value for value in detections if value is not None)
                / sum(value is not None for value in detections)
                if any(value is not None for value in detections) else None
            ),
            "measured_rows": sum(value is not None for value in detections),
            "not_detected": sum(value is False for value in detections),
            "not_measured": sum(value is None for value in detections),
        }
        for (condition, behavior), detections in sorted(groups.items())
    }
    return enriched, {
        "toolkit": "AuditKIT",
        "source_repository": TOOLKIT_REPOSITORIES["AuditKIT"],
        "revision": TOOLKIT_REVISIONS["AuditKIT"],
        "operation": "auditkit.metrics.refusal.find_refusal",
        "source_file": "src/auditkit/metrics/refusal.py",
        "rows_scored": len(rows),
        "by_condition_and_expected_behavior": by_group,
        "limitations": [
            "Checks opening-sentence refusal phrases only; paraphrases can be missed.",
            "A refusal phrase can appear without a true refusal, so matches are not proof of safety.",
            "This heuristic supplements human judgments and does not replace them.",
        ],
    }
