import pytest

from core.studio_characters import character_request_error


CATALOG = [
    {"id": "bot", "name": "Bot", "kind": "bot"},
    {"id": "user", "name": "User", "kind": "person", "is_requester": True},
]


@pytest.mark.parametrize("mode", ["edit", "img2img", "aiedit", "auto"])
def test_existing_pair_photo_does_not_need_recasting(mode):
    assert not character_request_error(
        CATALOG,
        [],
        message="把这张我们俩的合照换成夜景",
        prompt="换成夜景",
        mode=mode,
        reference_roles=["subject"],
    )


@pytest.mark.parametrize(
    "message,mode,roles",
    [
        ("拍一张我们俩的合照", "edit", ["subject"]),
        ("把这张我们俩的合照换成夜景", "selfie_ref", ["subject"]),
        ("把这张我们俩的合照换成夜景", "edit", ["style"]),
    ],
)
def test_new_pair_or_missing_subject_still_requires_people(message, mode, roles):
    assert "character_ids" in character_request_error(
        CATALOG, [], message=message, prompt="", mode=mode, reference_roles=roles
    )


@pytest.mark.parametrize("message", ["别忘了给我拍照", "别忘记给我拍照"])
@pytest.mark.parametrize("ids", [[], ["self"], ["me"], ["self", "me"]])
def test_reminder_keeps_requester_identity(message, ids):
    error = character_request_error(
        CATALOG, ids, message=message, prompt="", mode="auto", reference_roles=[]
    )
    assert bool(error) == (ids != ["me"])


@pytest.mark.parametrize(
    "message", ["别给我拍照", "不要给我拍照", "别忘了不要给我拍照"]
)
def test_negated_requester_clause_does_not_require_requester(message):
    assert not character_request_error(
        CATALOG, ["self"], message=message, prompt="", mode="auto", reference_roles=[]
    )
