from __future__ import annotations

from streamguard_bench.contracts import HarmOnset, NormalizedTrace, PrefixAnnotation
from streamguard_bench.data.build_subset import SubsetSpecification, build_balanced_subset


def _trace(index: int, split: str, label: str, length: int, onset: str | None = None):
    response = chr(65 + index % 26) * length
    harm_onset = None
    annotations = ()
    categories = ()
    if label == "unsafe":
        fraction = {"early": 0.2, "middle": 0.5, "late": 0.8}[onset]
        upper = int(length * fraction)
        lower = max(0, upper - 5)
        harm_onset = HarmOnset(
            lower, upper, lower, upper, 1, lower, upper, "fixture"
        )
        annotations = (
            PrefixAnnotation(0, upper, upper, 1, "unsafe", "unsafe", ("Violent",), (index,)),
        )
        categories = ("Violent", "Rare Category") if index == 0 else ("Violent",)
    return NormalizedTrace(
        trace_id=f"{split}-{label}-{onset}-{index:05d}",
        source_split=split,
        prompt=f"prompt-{split}-{label}-{index}",
        response=response,
        binary_label=label,
        original_label=label,
        harm_categories=categories,
        prefix_annotations=annotations,
        harm_onset=harm_onset,
        language="en",
        source_rows=(index,),
        dataset_revision="fixture",
        exclusion_reason=None,
    )


def _candidate_pool():
    traces = []
    index = 0
    for split, per_length in (("train", 100), ("val", 35)):
        for length in (10, 50, 100):
            for _ in range(per_length):
                traces.append(_trace(index, split, "safe", length))
                index += 1
    for split, counts in (
        ("train", {"early": 130, "middle": 70, "late": 70}),
        ("val", {"early": 45, "middle": 35, "late": 35}),
    ):
        for onset, count in counts.items():
            for _ in range(count):
                traces.append(_trace(index, split, "unsafe", 100, onset))
                index += 1
    return traces


def test_subset_is_exact_and_reproducible():
    pool = _candidate_pool()
    specification = SubsetSpecification()
    first, first_checks = build_balanced_subset(pool, specification)
    second, second_checks = build_balanced_subset(pool, specification)

    assert [trace.trace_id for trace in first] == [trace.trace_id for trace in second]
    assert len(first) == 500
    assert len({trace.trace_id for trace in first}) == 500
    assert (first_checks["status"] == "pass").all()
    assert (second_checks["status"] == "pass").all()
