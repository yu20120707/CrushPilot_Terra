from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SPEC_IDS = {
    *(f"SPEC-ADR-{index:03d}" for index in range(1, 13)),
    "SPEC-ARCH-001",
    *(f"SPEC-LLM-{index:03d}" for index in range(1, 4)),
    "SPEC-SKILL-001",
    "SPEC-SKILL-002",
    "SPEC-CTX-001",
    "SPEC-CTX-002",
    "SPEC-PLAN-001",
    "SPEC-PLAN-002",
    "SPEC-CORPUS-001",
    "SPEC-CORPUS-002",
    "SPEC-CORPUS-003",
    "SPEC-CORPUS-004",
    "SPEC-DB-001",
    "SPEC-FTS-001",
    "SPEC-FTS-002",
    "SPEC-VEC-001",
    "SPEC-VEC-002",
    "SPEC-FUSION-001",
    "SPEC-RERANK-001",
    "SPEC-RERANK-002",
    "SPEC-SELECT-001",
    "SPEC-SELECT-002",
    "SPEC-EVID-001",
    "SPEC-EVID-002",
    *(f"SPEC-PROMPT-POS-{index:03d}" for index in range(1, 10)),
    *(f"SPEC-PROMPT-BAN-{index:03d}" for index in range(1, 6)),
    *(f"SPEC-FAIL-{index:03d}" for index in range(1, 11)),
    "SPEC-OBS-001",
    "SPEC-OBS-002",
    "SPEC-MOD-001",
    "SPEC-AGENT-001",
    *(f"SPEC-PUB-{index:03d}" for index in range(1, 5)),
    "SPEC-TR-001",
    "SPEC-EVAL-001",
    "SPEC-EVAL-002",
    "SPEC-EVAL-003",
    "SPEC-TEST-002",
    *(f"SPEC-TEST-U-{index:03d}" for index in range(1, 12)),
    *(f"SPEC-TEST-E2E-{index:03d}" for index in range(1, 8)),
    *(f"SPEC-MIG-{index:03d}" for index in range(7)),
    *(f"SPEC-DONE-{index:03d}" for index in range(1, 13)),
    "SPEC-SUMMARY-001",
}
REQUIRED_TASK_IDS = {
    f"{index:02d}-{name}"
    for index, name in enumerate(
        (
            "source-inventory",
            "skill-runtime",
            "ingestion-and-chunking",
            "postgres-schema",
            "lexical-retrieval",
            "vector-retrieval",
            "fusion-and-rerank",
            "evidence-and-context",
            "agent-integration",
            "observability",
            "evaluation",
            "migration-cutover",
            "final-convergence",
        ),
        1,
    )
}
TASK_STATUSES = {"completed", "in_progress", "pending"}


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
    required_task_ids: set[str] | None = None,
) -> dict[str, Any]:
    trace = _load(traceability)
    requirements = trace.get("requirements")
    mapped: dict[str, dict[str, Any]] = {}
    invalid_requirements = int(trace.get("schema_version") != 2)
    duplicate_requirements = 0
    missing_evidence: list[str] = []
    if not isinstance(requirements, list):
        invalid_requirements += 1
        requirements = []
    for requirement in requirements:
        if not isinstance(requirement, dict) or not requirement.get("id"):
            invalid_requirements += 1
            continue
        requirement_id = str(requirement["id"])
        if requirement_id in mapped:
            duplicate_requirements += 1
        mapped[requirement_id] = requirement
        sections = requirement.get("source_sections")
        tasks = requirement.get("tasks")
        evidence = requirement.get("evidence", {})
        files = evidence.get("files", []) if isinstance(evidence, dict) else []
        if (
            not isinstance(sections, list)
            or not sections
            or any(not isinstance(section, str) or not section for section in sections)
            or len(sections) != len(set(sections))
            or not requirement.get("summary")
            or not isinstance(tasks, list)
            or not tasks
            or any(not isinstance(task, str) or not task for task in tasks)
            or len(tasks) != len(set(tasks))
            or not isinstance(files, list)
            or not files
            or any(not isinstance(file, str) or not file for file in files)
            or len(files) != len(set(files))
        ):
            invalid_requirements += 1
            continue
        root_resolved = root.resolve()
        evidence_paths = [(root / str(path)).resolve() for path in files]
        if any(
            not path.is_relative_to(root_resolved) or not path.is_file()
            for path in evidence_paths
        ):
            missing_evidence.append(requirement_id)

    task_records: dict[str, dict[str, Any]] = {}
    invalid_tasks: list[str] = []
    for status_path in sorted(tasks_dir.glob("*/status.yaml")):
        record = _load(status_path)
        task = status_path.parent.name
        task_records[task] = record
        spec_ids = record.get("spec_ids")
        evidence = record.get("evidence")
        if (
            record.get("task") != task
            or record.get("status") not in TASK_STATUSES
            or not isinstance(spec_ids, list)
            or any(not isinstance(spec_id, str) or not spec_id for spec_id in spec_ids)
            or len(spec_ids) != len(set(spec_ids))
            or (
                record.get("status") == "completed"
                and (
                    not isinstance(evidence, list)
                    or not evidence
                    or any(not isinstance(item, str) or not item for item in evidence)
                    or len(evidence) != len(set(evidence))
                )
            )
            or (
                record.get("status") == "in_progress"
                and not record.get("remaining")
            )
        ):
            invalid_tasks.append(task)
    expected_ids = set(
        REQUIRED_SPEC_IDS if required_spec_ids is None else required_spec_ids
    )
    task_spec_ids = {
        str(spec_id)
        for record in task_records.values()
        for spec_id in record.get("spec_ids", [])
    }
    unmapped = sorted(expected_ids - mapped.keys())
    unknown_spec_ids = sorted((mapped.keys() | task_spec_ids) - expected_ids)
    expected_tasks = set(
        required_task_ids
        if required_task_ids is not None
        else REQUIRED_TASK_IDS if required_spec_ids is None else task_records
    )
    missing_tasks = sorted(expected_tasks - task_records.keys())
    unknown_tasks = sorted(task_records.keys() - expected_tasks)

    unverified: set[str] = set(invalid_tasks) | set(missing_tasks) | set(unknown_tasks)
    referenced_tasks = {
        str(task)
        for requirement in mapped.values()
        for task in requirement.get("tasks", [])
    }
    unverified.update(referenced_tasks ^ expected_tasks)
    for task in sorted(referenced_tasks):
        record = task_records.get(task)
        if not record or record.get("task") != task or record.get("status") != "completed":
            unverified.add(task)
            continue
        mapped_ids = {
            requirement_id
            for requirement_id, requirement in mapped.items()
            if task in requirement.get("tasks", [])
        }
        if mapped_ids != set(record.get("spec_ids", [])):
            unverified.add(task)
            continue
        evidence = record.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            unverified.add(task)
            continue
        root_resolved = root.resolve()
        evidence_paths = [(root / str(path)).resolve() for path in evidence]
        if any(
            not path.is_relative_to(root_resolved) or not path.is_file()
            for path in evidence_paths
        ):
            unverified.add(task)

    return {
        "unmapped_requirements": (
            len(unmapped)
            + len(unknown_spec_ids)
            + invalid_requirements
            + duplicate_requirements
            + len(missing_evidence)
        ),
        "unverified_tasks": len(unverified),
        "details": {
            "unmapped_spec_ids": unmapped,
            "unknown_spec_ids": unknown_spec_ids,
            "missing_evidence": sorted(missing_evidence),
            "unverified_tasks": sorted(unverified),
            "invalid_tasks": sorted(invalid_tasks),
            "missing_tasks": missing_tasks,
            "unknown_tasks": unknown_tasks,
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
