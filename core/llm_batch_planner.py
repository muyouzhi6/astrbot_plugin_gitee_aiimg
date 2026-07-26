from __future__ import annotations

import json
from dataclasses import dataclass


SUPPORTED_ASPECT_RATIOS = (
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "3:2",
    "2:3",
    "5:4",
    "4:5",
    "21:9",
)


@dataclass(slots=True)
class PlannedPromptItem:
    title: str
    prompt: str
    variation_focus: list[str]
    aspect_ratio: str


def build_batch_planning_prompt(
    *,
    mode: str,
    user_prompt: str,
    count: int,
    fixed_aspect_ratio: str | None = None,
) -> str:
    fixed_ratio = str(fixed_aspect_ratio or "").strip()
    if fixed_ratio:
        ratio_rule = f"每个元素的 aspect_ratio 必须固定为 {fixed_ratio}。"
    elif int(count) > 1:
        ratio_rule = (
            "根据每张图的构图独立选择 aspect_ratio，整组至少使用两种比例；"
            "人像/自拍优先 3:4、4:5、9:16，横向场景优先 4:3、3:2、16:9，"
            "只有确实适合方形构图时才使用 1:1。"
        )
    else:
        ratio_rule = (
            "根据构图主动选择 aspect_ratio；人像/自拍优先 3:4、4:5、9:16，"
            "横向场景优先 4:3、3:2、16:9，只有确实适合时才使用 1:1。"
        )
    return (
        "你要为一个图片批量任务规划一组彼此不重复、但整体都满足要求的提示词。\n"
        f"模式: {mode}\n"
        f"目标数量: {count}\n"
        "要求:\n"
        "1. 每条都必须符合用户总要求。\n"
        "2. 整组之间不能只是同一句话换个近义词，必须在姿势、角度、表情、构图、动作、服装细节或氛围上形成明确区分。\n"
        "3. 不要输出解释，不要输出 markdown，不要输出代码块，只输出 JSON 数组。\n"
        '4. 每个元素格式必须是 {"title": "...", "prompt": "...", "variation_focus": ["..."], "aspect_ratio": "3:4"}。\n'
        "5. aspect_ratio 只能从 1:1、16:9、9:16、4:3、3:4、3:2、2:3、5:4、4:5、21:9 中选择。\n"
        f"6. {ratio_rule}\n"
        "7. title 要简短，prompt 要可直接用于图片生成或改图。\n\n"
        f"用户总要求:\n{str(user_prompt or '').strip()}"
    )


def _strip_code_fence(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return raw


def _normalize_compare_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def parse_planned_prompt_items(text: str) -> list[PlannedPromptItem]:
    raw = _strip_code_fence(text)
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("planner output must be a JSON array")

    out: list[PlannedPromptItem] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each planner item must be an object")
        title = str(item.get("title") or "").strip()
        prompt = str(item.get("prompt") or "").strip()
        aspect_ratio = str(item.get("aspect_ratio") or "").strip()
        variation_focus_raw = item.get("variation_focus") or []
        if isinstance(variation_focus_raw, list):
            variation_focus = [
                str(x).strip() for x in variation_focus_raw if str(x).strip()
            ]
        else:
            variation_focus = []
        out.append(
            PlannedPromptItem(
                title=title,
                prompt=prompt,
                variation_focus=variation_focus,
                aspect_ratio=aspect_ratio,
            )
        )
    return out


def validate_planned_prompt_items(
    items: list[PlannedPromptItem],
    *,
    expected_count: int,
    fixed_aspect_ratio: str | None = None,
) -> str | None:
    if len(items) != int(expected_count):
        return f"expected {expected_count} items, got {len(items)}"

    seen_titles: set[str] = set()
    seen_prompts: set[str] = set()
    used_ratios: set[str] = set()
    fixed_ratio = str(fixed_aspect_ratio or "").strip()
    for idx, item in enumerate(items, start=1):
        if not item.title:
            return f"item {idx} missing title"
        if not item.prompt:
            return f"item {idx} missing prompt"
        if item.aspect_ratio not in SUPPORTED_ASPECT_RATIOS:
            return f"item {idx} has invalid aspect_ratio"
        if fixed_ratio and item.aspect_ratio != fixed_ratio:
            return f"item {idx} must use aspect_ratio {fixed_ratio}"

        normalized_title = _normalize_compare_text(item.title)
        normalized_prompt = _normalize_compare_text(item.prompt)
        if normalized_title in seen_titles:
            return f"item {idx} duplicated title"
        if normalized_prompt in seen_prompts:
            return f"item {idx} duplicated prompt"
        seen_titles.add(normalized_title)
        seen_prompts.add(normalized_prompt)
        used_ratios.add(item.aspect_ratio)

    if not fixed_ratio and int(expected_count) > 1 and len(used_ratios) < 2:
        return "batch planner must use at least two aspect ratios"

    return None
