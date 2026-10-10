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


@pytest.mark.parametrize(
    "message,ids",
    [
        ("你和猫一起拍一张", ["self"]),
        ("请你跟一只小猫合影", ["self"]),
        ("你与猫咪同框", ["self"]),
        ("你和狗狗拍一张合照", ["self"]),
        ("我和猫一起拍一张", ["me"]),
        ("User和猫一起拍一张", ["me"]),
        ("你和猫一起拍2张", ["self"]),
    ],
)
@pytest.mark.parametrize("roles", [[], ["object"]])
def test_pet_companion_does_not_require_second_person(message, ids, roles):
    assert not character_request_error(
        CATALOG, ids, message=message, prompt="", mode="edit", reference_roles=roles
    )


@pytest.mark.parametrize("pet_name", ["猫", "小猫", "猫咪", "狗狗"])
@pytest.mark.parametrize("roles", [[], ["object"]])
def test_registered_pet_name_still_requires_named_person(pet_name, roles):
    catalog = CATALOG + [{"id": "friend", "name": pet_name, "kind": "person"}]
    args = dict(
        message=f"你和{pet_name}一起拍一张",
        prompt="",
        mode="edit",
        reference_roles=roles,
    )
    assert character_request_error(catalog, ["self"], **args)
    assert not character_request_error(catalog, ["self", "friend"], **args)


def test_requester_with_pet_still_requires_requester_identity():
    assert character_request_error(
        CATALOG,
        ["self"],
        message="我和猫一起拍一张",
        prompt="",
        mode="edit",
        reference_roles=[],
    )


@pytest.mark.parametrize(
    "message",
    [
        "你和我一起拍一张",
        "你和User一起拍一张",
        "我们俩和猫一起拍一张",
        "你和User和猫一起拍一张",
        "你和朋友一起拍一张",
        "你和猫娘一起拍一张",
        "拍一张双人合照",
        "你和猫一起拍一张，再拍一张我们俩的合照",
        "你和猫一起拍一张，再和User合影",
        "你和猫一起拍一张，再和朋友合影",
    ],
)
def test_pet_exception_does_not_weaken_human_group_validation(message):
    assert character_request_error(
        CATALOG, ["self"], message=message, prompt="", mode="edit", reference_roles=[]
    )
