from streamguard_bench.contracts import GuardDecision


def test_guard_decision_keeps_numeric_context() -> None:
    decision = GuardDecision("safe", 0.1, 10, 10, 2.5)
    assert decision.generated_tokens == decision.checked_tokens == 10
