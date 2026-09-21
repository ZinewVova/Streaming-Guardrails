import pytest

from streamguard_bench.guards import GuardOutputError, parse_guard_output


@pytest.mark.parametrize(
    "value",
    [{}, {"risk_level": None}, {"risk_level": []}, {"risk_level": ""}, {"risk_level": "mystery"}],
)
def test_invalid_risk_level_is_protocol_error(value):
    with pytest.raises(GuardOutputError):
        parse_guard_output(value)


@pytest.mark.parametrize("label", ["Safe", "Controversial", "Unsafe"])
def test_documented_labels_are_parsed(label):
    assert parse_guard_output({"risk_level": [label]})["risk_label"] == label.lower()
