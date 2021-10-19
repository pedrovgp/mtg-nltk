from mtgnlp import config
import pytest
import re


@pytest.fixture
def get_abilities_from_rules():
    with open(config.ROOT_DIR.joinpath("rules.txt"), "r", encoding="latin-1") as f:
        comprules = "\n".join(f.readlines())

    kw_abilities_pat = r"702\.\d+\. ([A-Za-z ]+)"
    abilities = re.findall(kw_abilities_pat, comprules, re.IGNORECASE)
    abilities.pop(0)  # Its just the rulings
    abilities.sort()
    return [a.lower() for a in abilities]


@pytest.fixture
def list_some_abilities():
    return [
        "first strike",
        "double strike",
        "lifelink",
        "deathtouch",
        "absorb",
        "affinity",
        "afflict",
        "afterlife",
        "aftermath",
        "amplify",
    ]


class TestRulesFile:
    def test_first_strike_in_abilities(
        self, list_some_abilities, get_abilities_from_rules
    ):
        for a in list_some_abilities:
            assert a in get_abilities_from_rules

    def test_most_no_in_abilities(self, get_abilities_from_rules):
        """Rule 702.1 starts with "most", but it is a ruling, not an ability"""
        assert "most" not in get_abilities_from_rules

    def test_abilities_len_gte(self, get_abilities_from_rules, length=30):
        assert len(get_abilities_from_rules) >= length
