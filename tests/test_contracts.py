from streamguard_bench.contracts import HarmOnset


def test_harm_onset_keeps_interval_coordinates() -> None:
    onset = HarmOnset(2, 5, 2, 6, 1, 2, 5, "fixture")
    assert onset.lower_character == 2
    assert onset.upper_character == 5
