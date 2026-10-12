"""Validation shared by image-provider retry settings."""


def parse_max_retries(value: object, default: int = 2) -> int:
    if value is None:
        return default
    if isinstance(value, str) and value.strip().isascii() and value.strip().isdecimal():
        value = int(value.strip())
    if type(value) is not int or value < 0:
        raise ValueError("max_retries must be a non-negative integer")
    return value
