"""Readiness-checked access to one compiled domain skill."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .compiler import DEFAULT_REQUIRED_SCENES, SkillCompilationError, validate_skill_semantics
from .loader import SkillLoadError, load_compiled_skill, load_skill_source, source_sha256
from .renderer import render_runtime_prompt


@dataclass(frozen=True)
class SkillReadiness:
    ready: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillRuntimeView:
    skill_id: str
    version: str
    source_sha256: str
    core_rules: tuple[dict[str, Any], ...]
    scene_policies: dict[str, dict[str, Any]]
    output_policy: dict[str, Any]
    prompt: str


class SkillRuntime:
    def __init__(self, skill_dir: Path | str, required_scenes: Iterable[str] = DEFAULT_REQUIRED_SCENES):
        self.skill_dir = Path(skill_dir)
        self.required_scenes = tuple(required_scenes)
        self._artifact: dict[str, Any] | None = None

    def readiness(self) -> SkillReadiness:
        try:
            artifact = load_compiled_skill(self.skill_dir)
            config = artifact["config"]
            source_config = load_skill_source(self.skill_dir)
            validate_skill_semantics(config, self.required_scenes)
            actual_hash = source_sha256(self.skill_dir / "skill.yaml")
            if artifact.get("source_sha256") != actual_hash:
                raise SkillLoadError("compiled skill source hash does not match skill.yaml")
            if config != source_config:
                raise SkillLoadError("compiled skill config does not match skill.yaml")
            metadata = config["metadata"]
            if artifact.get("skill_id") != metadata["id"] or artifact.get("skill_version") != metadata["version"]:
                raise SkillLoadError("compiled skill metadata does not match its config")
            self._artifact = artifact
            return SkillReadiness(True)
        except (OSError, KeyError, TypeError, SkillLoadError, SkillCompilationError) as exc:
            self._artifact = None
            return SkillReadiness(False, (str(exc),))

    def load(self) -> "SkillRuntime":
        status = self.readiness()
        if not status.ready:
            raise SkillLoadError("; ".join(status.errors))
        return self

    @property
    def source_hash(self) -> str:
        return self._require_artifact()["source_sha256"]

    @property
    def version(self) -> str:
        return self._require_artifact()["skill_version"]

    def view(self, active_scenes: Iterable[str] = ()) -> SkillRuntimeView:
        artifact = self._require_artifact()
        config = artifact["config"]
        scene_ids = tuple(dict.fromkeys(active_scenes))
        selected = {scene_id: config["scene_policies"][scene_id] for scene_id in scene_ids}
        return SkillRuntimeView(
            skill_id=artifact["skill_id"],
            version=artifact["skill_version"],
            source_sha256=artifact["source_sha256"],
            core_rules=tuple(config["core_rules"]),
            scene_policies=selected,
            output_policy=config["output_policy"],
            prompt=render_runtime_prompt(config, scene_ids),
        )

    def _require_artifact(self) -> dict[str, Any]:
        if self._artifact is None:
            raise SkillLoadError("skill runtime is not loaded; call load() after compilation")
        return self._artifact
