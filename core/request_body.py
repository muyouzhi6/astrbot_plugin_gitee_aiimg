"""Merge provider options without sharing mutable request state."""

from copy import deepcopy
import json


def merge_request_body(payload: dict, extra_body: dict | None) -> dict:
    result = deepcopy(payload)
    if extra_body is None:
        return result
    if not isinstance(extra_body, dict):
        raise ValueError("extra_body must be a JSON object")
    for key, value in extra_body.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_request_body(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def request_field(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False)
