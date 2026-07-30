"""Load and validate skill source and compiled artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class SkillLoadError(ValueError):
    """Raised when a skill file cannot be safely loaded."""


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_skill_source(skill_dir: Path | str) -> dict[str, Any]:
    directory = Path(skill_dir)
    source_path = directory / "skill.yaml"
    schema_path = directory / "schema.json"
    if not source_path.is_file() or not schema_path.is_file():
        raise SkillLoadError("skill.yaml and schema.json are required")

    try:
        import yaml
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise SkillLoadError("PyYAML and jsonschema are required to load skill.yaml") from exc

    try:
        source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise SkillLoadError(f"invalid skill source: {exc}") from exc
    if not isinstance(source, dict):
        raise SkillLoadError("skill.yaml root must be an object")

    errors = sorted(Draft202012Validator(schema).iter_errors(source), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise SkillLoadError(f"skill schema validation failed at {location}: {error.message}")
    return source


def load_compiled_skill(skill_dir: Path | str) -> dict[str, Any]:
    artifact_path = Path(skill_dir) / "skill.compiled.json"
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SkillLoadError("compiled skill artifact is missing") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SkillLoadError(f"compiled skill artifact is invalid: {exc}") from exc
    if not isinstance(artifact, dict) or not isinstance(artifact.get("config"), dict):
        raise SkillLoadError("compiled skill artifact has an invalid shape")
    return artifact
