from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SPEC_IDS = {
    *(f"SPEC-ADR-{index:03d}" for index in range(1, 13)),
    "SPEC-TR-001",
    "SPEC-EVAL-001",
    "SPEC-EVAL-002",
    "SPEC-EVAL-003",
    "SPEC-P0-001",
}


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def verify(
    traceability: Path,
    tasks_dir: Path,
    root: Path = ROOT,
    required_spec_ids: set[str] | None = None,
) -> dict[str, Any]:
    trace = _load(traceability)
    requirements = trace.get("requirements", [])
    mapped: dict[str, dict[str, Any]] = {}
    invalid_requirements = 0
    duplicate_requirements = 0
    for requirement in requirements if isinstance(requirements, list) else []:
        if not isinstance(requirement, dict) or not requirement.get("id"):
            invalid_requirements += 1
            continue
        requirement_id = str(requirement["id"])
        if requirement_id in mapped:
            duplicate_requirements += 1
        mapped[requirement_id] = requirement
        if not requirement.get("source") or not requirement.get("tasks"):
            invalid_requirements += 1

    task_records: dict[str, dict[str, Any]] = {}
    for status_path in sorted(tasks_dir.glob("*/status.yaml")):
        record = _load(status_path)
        task_records[status_path.parent.name] = record
    expected_ids = set(required_spec_ids or REQUIRED_SPEC_IDS) | {
        str(spec_id)
        for record in task_records.values()
        for spec_id in record.get("spec_ids", [])
    }
    unmapped = sorted(expected_ids - mapped.keys())

    unverified: list[str] = []
    referenced_tasks = {
        str(task)
        for requirement in mapped.values()
        for task in requirement.get("tasks", [])
    }
    for task in sorted(referenced_tasks):
        record = task_records.get(task)
        if not record or record.get("task") != task or record.get("status") != "completed":
            unverified.append(task)
            continue
        mapped_ids = {
            requirement_id
            for requirement_id, requirement in mapped.items()
            if task in requirement.get("tasks", [])
        }
        if not mapped_ids.issubset(set(record.get("spec_ids", []))):
            unverified.append(task)
            continue
        evidence = record.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            unverified.append(task)
            continue
        root_resolved = root.resolve()
        evidence_paths = [(root / str(path)).resolve() for path in evidence]
        if any(
            not path.is_relative_to(root_resolved) or not path.is_file()
            for path in evidence_paths
        ):
            unverified.append(task)

    return {
        "unmapped_requirements": len(unmapped) + invalid_requirements + duplicate_requirements,
        "unverified_tasks": len(unverified),
        "details": {
            "unmapped_spec_ids": unmapped,
            "unverified_tasks": unverified,
            "invalid_requirements": invalid_requirements,
            "duplicate_requirements": duplicate_requirements,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traceability",
        type=Path,
        default=ROOT / "trellis/retrieval-system-refactor/traceability.yaml",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=ROOT / "trellis/retrieval-system-refactor/tasks",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = verify(args.traceability, args.tasks_dir, args.root)
    except (OSError, ValueError, yaml.YAMLError) as error:
        result = {
            "unmapped_requirements": 1,
            "unverified_tasks": 1,
            "details": {"error": str(error)},
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    raise SystemExit(
        1
        if result["unmapped_requirements"] or result["unverified_tasks"]
        else 0
    )


if __name__ == "__main__":
    main()
