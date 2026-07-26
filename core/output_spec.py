from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from math import gcd

from PIL import Image as PILImage

_EXACT_SIZE_RE = re.compile(r"(\d{2,5})[xXx](\d{2,5})")
_ASPECT_RATIO_RE = re.compile(r"(\d{1,4}):(\d{1,4})")
_RESOLUTION_RE = re.compile(r"[1-9]\d*[kK]")
_PROMPT_EXACT_SIZE_RE = re.compile(
    r"(?<!\d)(\d{2,5})\s*[xX×]\s*(\d{2,5})(?!\d)"
)
_PROMPT_ASPECT_RATIO_RE = re.compile(r"(?<!\d)(\d{1,4})\s*:\s*(\d{1,4})(?!\d)")
_PROMPT_RESOLUTION_RE = re.compile(r"(?<![A-Za-z0-9])([124][kK])(?![A-Za-z0-9])")

COMMON_ASPECT_RATIOS = (
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


def normalize_exact_size(value: str | None) -> str | None:
    text = str(value or "").strip().replace("×", "x")
    match = _EXACT_SIZE_RE.fullmatch(text)
    if not match:
        return None
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    return f"{width}x{height}"


def normalize_aspect_ratio(value: str | None) -> str | None:
    match = _ASPECT_RATIO_RE.fullmatch(str(value or "").strip())
    if not match:
        return None
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    divisor = gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def normalize_resolution(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if _RESOLUTION_RE.fullmatch(text):
        return text.upper()
    return text


@dataclass(frozen=True, slots=True)
class OutputIntent:
    exact_size: str | None = None
    aspect_ratio: str | None = None
    resolution: str | None = None

    def __post_init__(self) -> None:
        exact_size = normalize_exact_size(self.exact_size)
        aspect_ratio = normalize_aspect_ratio(self.aspect_ratio)
        resolution = normalize_resolution(self.resolution)
        if self.exact_size and not exact_size:
            raise ValueError(f"invalid exact size: {self.exact_size}")
        if self.aspect_ratio and not aspect_ratio:
            raise ValueError(f"invalid aspect ratio: {self.aspect_ratio}")
        if exact_size and (aspect_ratio or resolution):
            raise ValueError("exact size cannot be combined with aspect ratio or resolution")
        object.__setattr__(self, "exact_size", exact_size)
        object.__setattr__(self, "aspect_ratio", aspect_ratio)
        object.__setattr__(self, "resolution", resolution)

    @property
    def is_empty(self) -> bool:
        return not (self.exact_size or self.aspect_ratio or self.resolution)

    def to_legacy_kwargs(self) -> dict[str, str]:
        if self.exact_size:
            return {"size": self.exact_size}
        if self.resolution:
            return {"resolution": self.resolution}
        return {}


def _classify_control_token(token: str) -> tuple[str, str] | None:
    exact_size = normalize_exact_size(token)
    if exact_size:
        return "exact_size", exact_size
    aspect_ratio = normalize_aspect_ratio(token)
    if aspect_ratio:
        return "aspect_ratio", aspect_ratio
    if _RESOLUTION_RE.fullmatch(str(token or "").strip()):
        return "resolution", str(token).strip().upper()
    return None


def is_output_control_token(token: str) -> bool:
    return _classify_control_token(token) is not None


def parse_output_intent(
    output: str | None, *, allow_legacy_resolution: bool = True
) -> OutputIntent:
    raw = str(output or "").strip()
    if not raw:
        return OutputIntent()

    tokens = [item for item in re.split(r"(?:\s*\+\s*|\s*,\s*|\s+)", raw) if item]
    values: dict[str, str] = {}
    for token in tokens:
        classified = _classify_control_token(token)
        if classified is None:
            if allow_legacy_resolution and len(tokens) == 1:
                return OutputIntent(resolution=token)
            raise ValueError(f"unsupported output token: {token}")
        kind, value = classified
        previous = values.get(kind)
        if previous and previous != value:
            raise ValueError(f"conflicting {kind}: {previous} and {value}")
        values[kind] = value

    return OutputIntent(
        exact_size=values.get("exact_size"),
        aspect_ratio=values.get("aspect_ratio"),
        resolution=values.get("resolution"),
    )


def output_intent_from_legacy(
    size: str | None, resolution: str | None
) -> OutputIntent:
    if str(size or "").strip():
        return OutputIntent(exact_size=str(size).strip())
    if str(resolution or "").strip():
        return OutputIntent(resolution=str(resolution).strip())
    return OutputIntent()


def merge_output_intents(*intents: OutputIntent | None) -> OutputIntent:
    aspect_ratio: str | None = None
    resolution: str | None = None
    for intent in intents:
        if intent is None or intent.is_empty:
            continue
        if intent.exact_size:
            if not aspect_ratio and not resolution:
                return intent
            if aspect_ratio is None:
                aspect_ratio = aspect_ratio_from_size(intent.exact_size)
            if resolution is None:
                resolution = resolution_from_size(intent.exact_size)
            continue
        if aspect_ratio is None and intent.aspect_ratio:
            aspect_ratio = intent.aspect_ratio
        if resolution is None and intent.resolution:
            resolution = intent.resolution
    return OutputIntent(aspect_ratio=aspect_ratio, resolution=resolution)


def format_output_intent(intent: OutputIntent | None) -> str:
    if intent is None or intent.is_empty:
        return ""
    if intent.exact_size:
        return intent.exact_size
    return " ".join(
        item for item in (intent.aspect_ratio, intent.resolution) if item
    )


def extract_output_intent_from_prompt(text: str | None) -> OutputIntent:
    """Extract explicit image controls embedded in a natural-language prompt."""
    raw = str(text or "")
    if not raw.strip():
        return OutputIntent()

    exact_matches = list(_PROMPT_EXACT_SIZE_RE.finditer(raw))
    if exact_matches:
        match = exact_matches[-1]
        return OutputIntent(
            exact_size=normalize_exact_size(
                f"{match.group(1)}x{match.group(2)}"
            )
        )

    aspect_ratio: str | None = None
    for match in _PROMPT_ASPECT_RATIO_RE.finditer(raw):
        candidate = normalize_aspect_ratio(f"{match.group(1)}:{match.group(2)}")
        if candidate in COMMON_ASPECT_RATIOS:
            aspect_ratio = candidate

    resolution_matches = list(_PROMPT_RESOLUTION_RE.finditer(raw))
    resolution = resolution_matches[-1].group(1).upper() if resolution_matches else None
    return OutputIntent(aspect_ratio=aspect_ratio, resolution=resolution)


def resolve_llm_output_intent(
    prompt: str | None,
    *,
    output: str | None = None,
    aspect_ratio: str | None = None,
    resolution: str | None = None,
) -> OutputIntent:
    """Resolve LLM fields without letting guessed defaults override the prompt."""
    auto_values = {"", "auto", "default", "\u9ed8\u8ba4"}

    structured_tokens: list[str] = []
    for value in (aspect_ratio, resolution):
        text = str(value or "").strip()
        if text.lower() not in auto_values:
            structured_tokens.append(text)

    output_text = str(output or "").strip()
    output_intent = (
        parse_output_intent(output_text)
        if output_text.lower() not in auto_values
        else OutputIntent()
    )
    structured_intent = (
        parse_output_intent(
            " ".join(structured_tokens), allow_legacy_resolution=False
        )
        if structured_tokens
        else OutputIntent()
    )
    return merge_output_intents(
        extract_output_intent_from_prompt(prompt),
        structured_intent,
        output_intent,
    )


def split_prompt_output_suffix(text: str | None) -> tuple[str, OutputIntent]:
    raw = str(text or "").strip()
    if not raw:
        return "", OutputIntent()

    matches = list(re.finditer(r"\S+", raw))
    controls: list[str] = []
    suffix_start = len(raw)
    for match in reversed(matches):
        if len(controls) == 2:
            break
        token = match.group(0)
        if not is_output_control_token(token):
            break
        controls.insert(0, token)
        suffix_start = match.start()

    if not controls:
        return raw, OutputIntent()
    return raw[:suffix_start].rstrip(), parse_output_intent(
        " ".join(controls), allow_legacy_resolution=False
    )


def aspect_ratio_from_size(size: str | None) -> str | None:
    normalized = normalize_exact_size(size)
    if not normalized:
        return None
    width_text, height_text = normalized.split("x", 1)
    return normalize_aspect_ratio(f"{width_text}:{height_text}")


def resolution_from_size(size: str | None) -> str | None:
    normalized = normalize_exact_size(size)
    if not normalized:
        return None
    width_text, height_text = normalized.split("x", 1)
    longest_edge = max(int(width_text), int(height_text))
    if longest_edge <= 1024:
        return "1K"
    if longest_edge <= 2048:
        return "2K"
    return "4K"


def detect_aspect_ratio_from_image(image_bytes: bytes) -> str | None:
    if not image_bytes:
        return None
    try:
        with PILImage.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None

    value = width / height
    return min(
        COMMON_ASPECT_RATIOS,
        key=lambda ratio: abs(
            math.log(
                value
                / (
                    int(ratio.split(":", 1)[0])
                    / int(ratio.split(":", 1)[1])
                )
            )
        ),
    )


def select_allowed_size(
    intent: OutputIntent,
    allowed_sizes: list[str] | tuple[str, ...],
    *,
    default_size: str | None = None,
) -> str | None:
    normalized_sizes = [
        normalized
        for item in allowed_sizes
        if (normalized := normalize_exact_size(item)) is not None
    ]
    if not normalized_sizes:
        return None
    if intent.exact_size:
        return intent.exact_size

    desired_ratio = intent.aspect_ratio or aspect_ratio_from_size(default_size)
    candidates = normalized_sizes
    if desired_ratio:
        same_ratio = [
            size for size in normalized_sizes if aspect_ratio_from_size(size) == desired_ratio
        ]
        if same_ratio:
            candidates = same_ratio
        elif intent.aspect_ratio:
            return None

    resolution_targets = {"1K": 1024, "2K": 2048, "4K": 4096}
    target_edge = resolution_targets.get(str(intent.resolution or "").upper())
    if target_edge is None:
        normalized_default = normalize_exact_size(default_size)
        if normalized_default:
            width_text, height_text = normalized_default.split("x", 1)
            target_edge = max(int(width_text), int(height_text))
        else:
            target_edge = 1024

    def distance(size: str) -> tuple[int, int]:
        width_text, height_text = size.split("x", 1)
        width = int(width_text)
        height = int(height_text)
        return abs(max(width, height) - target_edge), abs(width * height - target_edge**2)

    return min(candidates, key=distance)


def resolve_backend_output(backend: object, intent: OutputIntent) -> dict[str, str]:
    resolver = getattr(backend, "resolve_output_intent", None)
    if callable(resolver):
        resolved = resolver(intent)
        if not isinstance(resolved, dict):
            raise TypeError("resolve_output_intent() must return a dict")
        return {
            str(key): str(value)
            for key, value in resolved.items()
            if value is not None and str(value).strip()
        }
    return intent.to_legacy_kwargs()


def parse_output(output: str | None) -> tuple[str | None, str | None]:
    """Parse user output into (size, resolution).

    size: "2048x2048"
    resolution: "4K" / "2K" / "1K"
    """
    intent = parse_output_intent(output)
    if intent.exact_size:
        return intent.exact_size, None
    return None, intent.resolution
