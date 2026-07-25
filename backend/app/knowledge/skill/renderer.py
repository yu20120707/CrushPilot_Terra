"""Render human review and prompt views from structured skill data."""

from __future__ import annotations

from typing import Any, Iterable


def _bullets(values: Iterable[str]) -> str:
    return "\n".join(f"- {value}" for value in values)


def render_skill_markdown(config: dict[str, Any]) -> str:
    metadata = config["metadata"]
    sections = [
        "# 狗头军师 Skill",
        "",
        "> 此文件由 `skill.yaml` 自动生成，请勿手工编辑。",
        "",
        f"- ID: `{metadata['id']}`",
        f"- Version: `{metadata['version']}`",
        f"- Language: `{metadata['language']}`",
        "",
        "## Core Rules",
        "",
    ]
    for rule in config["core_rules"]:
        sections.extend((f"### {rule['id']} ({rule['priority']})", "", rule["instruction"], ""))
    sections.extend(("## Scene Policies", ""))
    for scene_id, policy in config["scene_policies"].items():
        sections.extend(
            (
                f"### {scene_id}",
                "",
                "**Required topics**",
                "",
                _bullets(policy["required_topics"]),
                "",
                "**Excluded topics**",
                "",
                _bullets(policy["excluded_topics"]) or "- None",
                "",
                "**Reasoning rules**",
                "",
                _bullets(policy["reasoning_rules"]),
                "",
            )
        )
    sections.extend(
        (
            "## Output Policy",
            "",
            "**Order**",
            "",
            _bullets(config["output_policy"]["order"]),
            "",
            "**Constraints**",
            "",
            _bullets(config["output_policy"]["constraints"]),
            "",
        )
    )
    return "\n".join(sections)


def render_runtime_prompt(config: dict[str, Any], active_scenes: Iterable[str] = ()) -> str:
    scene_ids = list(dict.fromkeys(active_scenes))
    unknown = [scene_id for scene_id in scene_ids if scene_id not in config["scene_policies"]]
    if unknown:
        raise KeyError(f"unknown scene policies: {', '.join(unknown)}")
    lines = ["Core Skill Rules:"]
    lines.extend(f"- [{rule['priority']}] {rule['instruction']}" for rule in config["core_rules"])
    for scene_id in scene_ids:
        policy = config["scene_policies"][scene_id]
        lines.extend(
            (
                f"Active Scene Policy: {scene_id}",
                f"- Required topics: {', '.join(policy['required_topics'])}",
                f"- Excluded topics: {', '.join(policy['excluded_topics']) or 'none'}",
            )
        )
        lines.extend(f"- {rule}" for rule in policy["reasoning_rules"])
    lines.append("Output Policy:")
    lines.append(f"- Order: {', '.join(config['output_policy']['order'])}")
    lines.extend(f"- {constraint}" for constraint in config["output_policy"]["constraints"])
    return "\n".join(lines)
