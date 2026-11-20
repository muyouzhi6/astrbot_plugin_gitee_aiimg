"""Face-preservation lines for person-reference (selfie / character) prompts.

Only person-reference flows call these helpers; plain image edits are untouched.
"""

from __future__ import annotations

import re

_START_LINE_RE = re.compile(
    r"^(?:"
    r"严格保持每位人物的脸、五官和发型与其对应身份参考图一致，不变。"
    r"|严格保持参考图\d+(?:、\d+)*中人物的脸、五官和发型不变。"
    r")$"
)
_END_LINE_RE = re.compile(
    r"^(?:"
    r"再次强调：每位人物的脸必须与其对应身份参考图一致。"
    r"|再次强调：人物的脸必须与参考图\d+(?:、\d+)*一致。"
    r")$"
)


def _label(ref_indices) -> str:
    nums = [int(i) for i in (ref_indices or [1])]
    return "参考图" + "、".join(str(n) for n in nums)


def face_lock_lines(ref_indices=(1,), *, multi_person: bool = False) -> tuple[str, str]:
    if multi_person:
        return (
            "严格保持每位人物的脸、五官和发型与其对应身份参考图一致，不变。",
            "再次强调：每位人物的脸必须与其对应身份参考图一致。",
        )
    label = _label(ref_indices)
    return (
        f"严格保持{label}中人物的脸、五官和发型不变。",
        f"再次强调：人物的脸必须与{label}一致。",
    )


def _split_lock(prompt: str) -> tuple[str | None, str, str | None]:
    lines = str(prompt or "").split("\n")
    start = end = None
    if lines and _START_LINE_RE.fullmatch(lines[0].strip()):
        start = lines.pop(0).strip()
    for i in range(len(lines) - 1, -1, -1):
        if _END_LINE_RE.fullmatch(lines[i].strip()):
            end = lines.pop(i).strip()
            break
    return start, "\n".join(lines).strip("\n"), end


def apply_face_lock(
    prompt: str, ref_indices=(1,), *, multi_person: bool = False
) -> str:
    """Wrap ``prompt`` with the face lines (idempotent)."""
    _, body, _ = _split_lock(prompt)
    start, end = face_lock_lines(ref_indices, multi_person=multi_person)
    return f"{start}\n{body}\n{end}"


def relock(prompt: str) -> str:
    """Move existing face lines back to the very start/end after text was appended."""
    start, body, end = _split_lock(prompt)
    if not start or not end:
        return prompt
    return f"{start}\n{body}\n{end}"


def strip_face_lock(prompt: str) -> str:
    """Remove standalone plugin face-lock wrapper lines while preserving prompt text."""
    _, body, _ = _split_lock(prompt)
    return body


def face_lock_for_cast(prompt: str, cast) -> str:
    cast = list(cast or [])
    if not cast:
        return prompt
    if len(cast) == 1:
        return apply_face_lock(prompt, cast[0].get("reference_indices") or [1])
    return apply_face_lock(prompt, multi_person=True)


def split_face_lock(prompt: str) -> tuple[str | None, str, str | None]:
    """Return (start_line, body, end_line); lines are None when absent."""
    start, body, end = _split_lock(prompt)
    if not start or not end:
        return None, str(prompt or ""), None
    return start, body, end
