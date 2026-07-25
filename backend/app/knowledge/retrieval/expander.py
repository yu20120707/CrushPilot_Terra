from __future__ import annotations


def expand_selected(
    selected: list[dict],
    by_id: dict[str, dict],
    include_parent: bool = True,
) -> list[dict]:
    expanded: list[dict] = []
    seen_context: set[str] = set()
    for candidate in selected:
        item = dict(candidate)
        context: list[dict] = []
        related_ids = []
        if include_parent:
            related_ids.append(candidate.get("parent_section_id"))
        related_ids.extend(
            (candidate.get("previous_chunk_id"), candidate.get("next_chunk_id"))
        )
        for related_id in related_ids:
            if not related_id or related_id in seen_context or related_id not in by_id:
                continue
            context.append(by_id[related_id])
            seen_context.add(related_id)
        item["expanded_context"] = context
        expanded.append(item)
    return expanded
