from pathlib import Path

import pandas as pd

from streamguard_bench.data.load_streamsafe import read_data_file
from streamguard_bench.data.validate_raw import validate_tables

FIXTURE = Path(__file__).parents[1] / "data" / "fixtures" / "synthetic_streamsafe_sample.json"


def test_fixture_validation_reports_duplicates_without_fatal_errors() -> None:
    report = validate_tables({"synthetic": read_data_file(FIXTURE)})
    assert not report.has_fatal
    assert "duplicate_rows" in set(report.to_frame()["code"])


def test_empty_table_is_fatal() -> None:
    report = validate_tables({"empty": pd.DataFrame()})
    assert report.has_fatal


def test_table_without_text_columns_is_fatal() -> None:
    report = validate_tables({"numbers": pd.DataFrame({"value": [1, 2, 3]})})
    assert report.has_fatal
    assert "no_text_candidate" in set(report.to_frame()["code"])
