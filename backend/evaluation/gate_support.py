from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATASET = Path(__file__).with_name("golden_dataset_v1.jsonl")
ROOT = Path(__file__).parents[2]
TRELIS = ROOT / "trellis" / "retrieval-system-refactor"


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def prepare_outputs(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def publish_outputs(
    results_path: Path,
    predictions_path: Path,
    partial_path: Path,
    results: dict[str, Any],
    predictions: dict[str, Any],
) -> None:
    atomic_json(predictions_path, predictions)
    atomic_json(results_path, results)  # Completion marker: publish last.
    partial_path.unlink(missing_ok=True)


def build_runtime_state(query: str, case_id: str) -> dict:
    return {
        "user_message": query,
        "device_id": "00000000-0000-0000-0000-000000000001",
        "conversation_id": f"golden-{case_id}",
        "relationship_id": None,
        "conversation_summary": None,
        "relationship_facts": [],
        "user_preferences": [],
        "messages": [{"role": "user", "content": query}],
    }
