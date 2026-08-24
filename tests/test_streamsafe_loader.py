from pathlib import Path

import pandas as pd

from streamguard_bench.data.load_streamsafe import load_local_snapshot, read_data_file

FIXTURE = Path(__file__).parents[1] / "data" / "fixtures" / "synthetic_streamsafe_sample.json"


def test_read_json_fixture_preserves_text() -> None:
    frame = read_data_file(FIXTURE)
    assert len(frame) == 12
    assert frame.loc[0, "response"] == (
        "Rainbows form when light is refracted and reflected in water droplets."
    )


def test_local_snapshot_keeps_files_separate(tmp_path: Path) -> None:
    pd.DataFrame({"prompt": ["one"], "response": ["first"]}).to_json(
        tmp_path / "full.json", orient="records"
    )
    pd.DataFrame({"trace_id": ["t1"], "prefix": ["partial"], "label": ["safe"]}).to_csv(
        tmp_path / "partial.csv", index=False
    )
    tables, files = load_local_snapshot(tmp_path)
    assert len(tables) == 2
    assert {tuple(frame.columns) for frame in tables.values()} == {
        ("prompt", "response"),
        ("trace_id", "prefix", "label"),
    }
    assert len(files) == 2


def test_no_network_is_used_for_local_snapshot(tmp_path: Path) -> None:
    fixture_copy = tmp_path / FIXTURE.name
    fixture_copy.write_bytes(FIXTURE.read_bytes())
    tables, _ = load_local_snapshot(tmp_path)
    assert sum(len(frame) for frame in tables.values()) == 12


def test_generated_manifest_is_not_loaded_as_source_data(tmp_path: Path) -> None:
    pd.DataFrame({"prompt": ["one"], "response": ["first"]}).to_json(
        tmp_path / "full.json", orient="records"
    )
    (tmp_path / "manifest.json").write_text(
        '{"dataset": "synthetic", "tables": {}}', encoding="utf-8"
    )

    tables, files = load_local_snapshot(tmp_path)

    assert list(tables) == ["full"]
    assert files == {"full": "full.json"}
