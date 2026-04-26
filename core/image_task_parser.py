from __future__ import annotations

from dataclasses import dataclass


COMMAND_PREFIXES = "/!！.。．"
DRAW_COMMANDS = {"aiimg", "文生图"}
EDIT_COMMANDS = {"aiedit", "改图", "图生图", "修图"}
SELFIE_COMMANDS = {"自拍"}


@dataclass(slots=True)
class ImageTaskSpec:
    mode: str
    provider_id: str | None
    preset_name: str | None
    user_prompt: str
    effective_prompt: str
    source_command: str
    variant_title: str = ""


@dataclass(slots=True)
class ParsedImageRequest:
    batch_count: int
    spec: ImageTaskSpec


def strip_leading_command_prefix(text: str) -> str:
    s = str(text or "")
    if s and s[0] in COMMAND_PREFIXES:
        return s[1:]
    return s


def split_first_token_and_rest(text: str) -> tuple[str, str]:
    s = str(text or "").lstrip()
    if not s:
        return "", ""

    idx = 0
    while idx < len(s) and not s[idx].isspace():
        idx += 1

    token = s[:idx]
    rest = s[idx:]
    return token, rest.lstrip()


def parse_provider_override_prefix(
    text: str, known_provider_ids: set[str]
) -> tuple[str | None, str]:
    s = str(text or "").lstrip()
    if not s.startswith("@"):
        return None, s

    token, rest = split_first_token_and_rest(s)
    candidate = token.lstrip("@").strip()
    if candidate and candidate in known_provider_ids:
        return candidate, rest
    return None, s


def build_draw_preset_prompt(preset_prompt: str, extra_prompt: str) -> str:
    preset = str(preset_prompt or "").strip()
    extra = str(extra_prompt or "").strip()
    if not extra:
        return preset
    return f"{preset}\n\n补充要求：\n{extra}"


def build_edit_effective_prompt(preset_prompt: str, extra_prompt: str) -> str:
    preset = str(preset_prompt or "").strip()
    extra = str(extra_prompt or "").strip()
    if not extra:
        return preset
    return f"{preset}, {extra}"


def parse_batch_prefix(text: str) -> tuple[int, str]:
    body = strip_leading_command_prefix(text).lstrip()
    if not body.startswith("批量"):
        return 1, body

    rest = body[len("批量") :]
    idx = 0
    while idx < len(rest) and rest[idx].isspace():
        idx += 1

    digit_start = idx
    while idx < len(rest) and rest[idx].isdigit():
        idx += 1

    digits = rest[digit_start:idx]
    if not digits:
        return 1, body

    remaining = rest[idx:].lstrip()
    return max(1, int(digits)), remaining


def parse_image_request(
    text: str,
    *,
    draw_presets: dict[str, str] | None = None,
    edit_presets: dict[str, str] | None = None,
    known_provider_ids: set[str] | None = None,
) -> ParsedImageRequest | None:
    draw_presets = draw_presets or {}
    edit_presets = edit_presets or {}
    known_provider_ids = known_provider_ids or set()

    batch_count, body = parse_batch_prefix(text)
    command, rest = split_first_token_and_rest(body)
    if not command:
        return None

    if command in DRAW_COMMANDS:
        provider_id, content = parse_provider_override_prefix(rest, known_provider_ids)
        if command == "文生图":
            first_token, extra_prompt = split_first_token_and_rest(content)
            if first_token and first_token in draw_presets:
                user_prompt = extra_prompt.strip()
                effective_prompt = build_draw_preset_prompt(
                    draw_presets[first_token], user_prompt
                )
                spec = ImageTaskSpec(
                    mode="draw",
                    provider_id=provider_id,
                    preset_name=first_token,
                    user_prompt=user_prompt,
                    effective_prompt=effective_prompt,
                    source_command=command,
                )
                return ParsedImageRequest(batch_count=batch_count, spec=spec)

        user_prompt = content.strip()
        spec = ImageTaskSpec(
            mode="draw",
            provider_id=provider_id,
            preset_name=None,
            user_prompt=user_prompt,
            effective_prompt=user_prompt,
            source_command=command,
        )
        return ParsedImageRequest(batch_count=batch_count, spec=spec)

    if command in EDIT_COMMANDS:
        provider_id, content = parse_provider_override_prefix(rest, known_provider_ids)
        user_prompt = content.strip()
        spec = ImageTaskSpec(
            mode="edit",
            provider_id=provider_id,
            preset_name=None,
            user_prompt=user_prompt,
            effective_prompt=user_prompt,
            source_command=command,
        )
        return ParsedImageRequest(batch_count=batch_count, spec=spec)

    if command in SELFIE_COMMANDS:
        provider_id, content = parse_provider_override_prefix(rest, known_provider_ids)
        user_prompt = content.strip()
        spec = ImageTaskSpec(
            mode="selfie_ref",
            provider_id=provider_id,
            preset_name=None,
            user_prompt=user_prompt,
            effective_prompt=user_prompt,
            source_command=command,
        )
        return ParsedImageRequest(batch_count=batch_count, spec=spec)

    if command in edit_presets:
        user_prompt = rest.strip()
        spec = ImageTaskSpec(
            mode="edit",
            provider_id=None,
            preset_name=command,
            user_prompt=user_prompt,
            effective_prompt=build_edit_effective_prompt(
                edit_presets[command], user_prompt
            ),
            source_command="edit_preset",
        )
        return ParsedImageRequest(batch_count=batch_count, spec=spec)

    return None
