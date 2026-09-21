import json
from pathlib import Path

import pandas as pd
import pytest

from streamguard_bench.data import prepare_frame, unsafe_start_token, validate_source

FIXTURE = Path("data/fixtures/synthetic_singstreambench_sample.json")


class Tokenizer:
    def __call__(self, text, **kwargs):
        # Includes a two-character token so inside-token behavior is exercised.
        offsets = [(0, 2)] + [(i, i + 1) for i in range(2, len(text))]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def fixture_frame():
    return pd.DataFrame(json.loads(FIXTURE.read_text()))


def test_schema_safe_prefix_and_unicode_are_preserved():
    frame = fixture_frame()
    frame.loc[0, "Response"] = "Café ☕"
    frame.loc[0, "Safe_Prefix"] = "Café ☕"
    frame.loc[0, "Unsafe_Start_Index"] = len("Café ☕")
    report = validate_source(frame, expected_rows=None)
    prepared = prepare_frame(frame, Tokenizer(), expected_rows=None)
    assert report["passed"].all()
    assert prepared.loc[0, "response"] == "Café ☕"


def test_bad_safe_prefix_is_rejected():
    frame = fixture_frame()
    frame.loc[1, "Safe_Prefix"] = "wrong"
    report = validate_source(frame, expected_rows=None)
    assert not report.set_index("check").loc["safe_prefix_exact", "passed"]


@pytest.mark.parametrize(("onset", "expected"), [(0, 1), (2, 2), (1, 1)])
def test_onset_zero_boundary_and_inside_token(onset, expected):
    assert (
        unsafe_start_token(unsafe_start_character=onset, response="abcd", offsets=[(0, 2), (2, 4)])
        == expected
    )


def test_safe_onset_maps_to_end_sentinel():
    assert (
        unsafe_start_token(unsafe_start_character=4, response="abcd", offsets=[(0, 2), (2, 4)]) == 3
    )
