"""Compile the YAML source into deterministic runtime and review artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .loader import SkillLoadError, load_skill_source, source_sha256
from .renderer import render_skill_markdown

DEFAULT_REQUIRED_SCENES = ("explicit_rejection", "emotional_support")
_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")


class SkillCompilationError(ValueError):
    """Raised when structurally valid skill source is not runtime-ready."""


def validate_skill_semantics(config: dict[str, Any], required_scenes: Iterable[str] = DEFAULT_REQUIRED_SCENES) -> None:
    version = config["metadata"]["version"]
    if not _SEMVER.fullmatch(version):
        raise SkillCompilationError(f"skill version is not valid semantic versioning: {version}")
    if not config["core_rules"]:
        raise SkillCompilationError("core_rules must not be empty")
    missing = sorted(set(required_scenes) - config["scene_policies"].keys())
    if missing:
        raise SkillCompilationError(f"required scene policies are missing: {', '.join(missing)}")
    for scene_id, policy in config["scene_policies"].items():
        if not policy["required_topics"]:
            raise SkillCompilationError(f"scene policy {scene_id} has no required topics")
        if not policy["reasoning_rules"]:
            raise SkillCompilationError(f"scene policy {scene_id} has no reasoning rules")


def compile_skill(
    skill_dir: Path | str,
    required_scenes: Iterable[str] = DEFAULT_REQUIRED_SCENES,
) -> dict[str, Any]:
    directory = Path(skill_dir)
    try:
        config = load_skill_source(directory)
    except SkillLoadError as exc:
        raise SkillCompilationError(str(exc)) from exc
    validate_skill_semantics(config, required_scenes)
    artifact = {
        "artifact_version": 1,
        "skill_id": config["metadata"]["id"],
        "skill_version": config["metadata"]["version"],
        "source_file": "skill.yaml",
        "source_sha256": source_sha256(directory / "skill.yaml"),
        "config": config,
    }
    (directory / "skill.compiled.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "SKILL.md").write_text(render_skill_markdown(config), encoding="utf-8")
    return artifact
