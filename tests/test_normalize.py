import pytest

from athar.normalize import norm, normalize


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("بكرة", "بكره"),  # taa marbuta
        ("أحمد إبراهيم آمال", "احمد ابراهيم امال"),  # alef variants
        ("مستشفى", "مستشفي"),  # alef maksura
        ("فَكِّرْنِي", "فكرني"),  # diacritics
        ("فكـــرني", "فكرني"),  # tatweel
        ("فكرنييييي", "فكرني"),  # elongation
        ("الساعة ٧:٣٠", "الساعه 7:30"),  # Arabic-Indic digits
        ("الساعة ۸", "الساعه 8"),  # Extended Arabic-Indic digits
        ("الساعة7م", "الساعه 7 م"),  # digit/letter split
        ("Remind ME at 7PM", "remind me at 7 pm"),  # code-switching, lower-case
        ("بكرة؟  الساعة،  ٧", "بكره? الساعه, 7"),  # punctuation + whitespace
        ("ﻻ", "لا"),  # presentation-form ligature (NFKC)
    ],
)
def test_normalize(raw, expected):
    assert norm(raw) == expected


def test_doubled_letters_are_kept():
    assert norm("الله") == "الله"
    assert norm("مدّة") == "مده"


def test_offsets_map_back_to_original():
    raw = "فكرني بكرة أشتري عيش"
    n = normalize(raw)
    start = n.text.index("اشتري")
    assert n.source_text(start, len(n.text)) == "أشتري عيش"


def test_offsets_with_deletions():
    raw = "أشـــتَري"
    n = normalize(raw)
    assert n.text == "اشتري"
    assert n.source_text(0, len(n.text)) == raw
