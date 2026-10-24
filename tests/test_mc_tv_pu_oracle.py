import math
from fractions import Fraction

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

import mc_tv_pu_oracle as oracle_module
from mc_tv_pu_oracle import (
    IncompatibleAmbiguitySet,
    OracleInputError,
    compatibility_excess,
    compatibility_threshold,
    compatibility_threshold_exact,
    prepare_mc_tv_pu_oracle_exact,
    saturation_threshold,
    solve_fixed_t,
    solve_fixed_t_exact,
    solve_mc_tv_pu_oracle,
    solve_mc_tv_pu_oracle_exact,
    solve_mc_tv_pu_oracle_exact_prepared,
    solve_t_one_limit,
    solve_t_one_limit_exact,
)


@pytest.mark.parametrize(
    ("q", "p0", "s", "tau", "prior_lower", "prior_upper"),
    [
        (["1/2", "1/2"], ["4/5", "1/5"], [1, -1], "1/10", 0, 1),
        (
            ["1/4", "1/4", "1/2"],
            ["1/2", "1/4", "1/4"],
            [1, -1, 0],
            "1/2",
            "1/5",
            "3/4",
        ),
        (
            ["1/3", "1/3", "1/3"],
            [1, 0, 0],
            [1, 0, -1],
            1,
            0,
            1,
        ),
    ],
)
def test_prepared_exact_geometry_matches_public_api(
    q, p0, s, tau, prior_lower, prior_upper
) -> None:
    geometry = prepare_mc_tv_pu_oracle_exact(
        q, p0, tau, prior_lower=prior_lower, prior_upper=prior_upper
    )

    public = solve_mc_tv_pu_oracle_exact(
        q,
        p0,
        s,
        tau,
        prior_lower,
        prior_upper,
        absolute_gap="1/1000",
    )
    prepared = solve_mc_tv_pu_oracle_exact_prepared(
        geometry, s, absolute_gap="1/1000"
    )

    assert prepared.to_dict() == public.to_dict()


def test_precomputed_score_orders_preserve_fixed_t_result() -> None:
    q = (Fraction(1, 4),) * 4
    p0 = (Fraction(1, 2), Fraction(1, 4), Fraction(1, 4), Fraction(0))
    s = (Fraction(2), Fraction(-1), Fraction(2), Fraction(-3))
    tau = Fraction(2, 5)
    t = Fraction(2)

    original = oracle_module._solve_fixed_t_exact_prevalidated(q, p0, s, tau, t)
    optimized = oracle_module._solve_fixed_t_exact_prevalidated(
        q, p0, s, tau, t, oracle_module._score_orders(s)
    )

    assert optimized.to_dict() == original.to_dict()


def test_prepared_exact_solver_sorts_score_direction_once(monkeypatch) -> None:
    geometry = prepare_mc_tv_pu_oracle_exact(
        ["1/4"] * 4,
        ["1/2", "1/4", "1/4", 0],
        "2/5",
        prior_lower=0,
        prior_upper=1,
    )
    calls = 0
    original = oracle_module._score_orders

    def tracked_score_orders(s_exact):
        nonlocal calls
        calls += 1
        return original(s_exact)

    monkeypatch.setattr(oracle_module, "_score_orders", tracked_score_orders)
    result = solve_mc_tv_pu_oracle_exact_prepared(
        geometry,
        [2, -1, 2, -3],
        absolute_gap="1/1000",
        max_evaluations=128,
    )

    assert result.evaluations > 1
    assert calls == 1


def test_score_order_cache_reuses_exact_direction_across_tau() -> None:
    oracle_module._score_orders.cache_clear()
    first_geometry = prepare_mc_tv_pu_oracle_exact(
        ["1/4"] * 4,
        ["1/2", "1/4", "1/4", 0],
        "2/5",
        prior_lower=0,
        prior_upper=1,
    )
    second_geometry = prepare_mc_tv_pu_oracle_exact(
        ["1/4"] * 4,
        ["1/2", "1/4", "1/4", 0],
        "3/5",
        prior_lower=0,
        prior_upper=1,
    )
    direction = [2, -1, 2, -3]

    solve_mc_tv_pu_oracle_exact_prepared(
        first_geometry,
        direction,
        absolute_gap="1/1000",
        max_evaluations=128,
    )
    cached = solve_mc_tv_pu_oracle_exact_prepared(
        second_geometry,
        direction,
        absolute_gap="1/1000",
        max_evaluations=128,
    )
    cache_info = oracle_module._score_orders.cache_info()

    oracle_module._score_orders.cache_clear()
    uncached = solve_mc_tv_pu_oracle_exact_prepared(
        second_geometry,
        direction,
        absolute_gap="1/1000",
        max_evaluations=128,
    )
    oracle_module._score_orders.cache_clear()

    assert cache_info.misses == 1
    assert cache_info.hits == 1
    assert cache_info.currsize == 1
    assert cached.to_dict() == uncached.to_dict()


def test_score_order_cache_is_bounded() -> None:
    oracle_module._score_orders.cache_clear()
    try:
        for index in range(257):
            oracle_module._score_orders((Fraction(index),))
        cache_info = oracle_module._score_orders.cache_info()
        assert cache_info.maxsize == 256
        assert cache_info.currsize == 256
        assert cache_info.misses == 257
    finally:
        oracle_module._score_orders.cache_clear()


def test_value_only_fixed_t_matches_full_witness_path() -> None:
    rng = np.random.default_rng(20261022)
    for case in range(80):
        n = 8
        q = (Fraction(1, n),) * n
        counts = rng.integers(0, 8, size=n)
        if int(counts.sum()) == 0:
            counts[0] = 1
        total = int(counts.sum())
        p0 = tuple(Fraction(int(count), total) for count in counts)
        raw_scores = tuple(
            Fraction(int(value)) for value in rng.integers(-5, 6, size=n)
        )
        mean = sum(
            (mass * score for mass, score in zip(q, raw_scores, strict=True)),
            Fraction(0),
        )
        direction = tuple(score - mean for score in raw_scores)
        t = Fraction(int(rng.integers(1, 7)))
        mandatory_mass = oracle_module.compatibility_excess_exact(t, q, p0)
        remaining_mass = Fraction(1) - mandatory_mass
        tau = mandatory_mass + remaining_mass * Fraction(case % 5, 4)
        if tau > 1:
            tau = Fraction(1)
        score_orders = oracle_module._score_orders(direction)
        baseline_value = sum(
            (
                score * mass
                for score, mass in zip(direction, p0, strict=True)
            ),
            Fraction(0),
        )

        full = oracle_module._solve_fixed_t_exact_prevalidated(
            q, p0, direction, tau, t, score_orders
        )
        value_only = oracle_module._solve_fixed_t_exact_value_prevalidated(
            q,
            p0,
            direction,
            tau,
            t,
            score_orders,
            baseline_value,
        )

        assert value_only.t == full.t
        assert value_only.value == full.value


def test_prepared_outer_oracle_does_not_materialize_fixed_t_witness(
    monkeypatch,
) -> None:
    geometry = prepare_mc_tv_pu_oracle_exact(
        ["1/4"] * 4,
        ["1/2", "1/4", "1/4", 0],
        "2/5",
        prior_lower=0,
        prior_upper=1,
    )

    def reject_full_witness(*args, **kwargs):
        raise AssertionError("prepared outer oracle requested a full fixed-t witness")

    monkeypatch.setattr(
        oracle_module, "_solve_fixed_t_exact_prevalidated", reject_full_witness
    )
    result = solve_mc_tv_pu_oracle_exact_prepared(
        geometry,
        [2, -1, 2, -3],
        absolute_gap="1/1000",
        max_evaluations=128,
    )

    assert result.certified


def test_validation_rejects_non_centered_s() -> None:
    with pytest.raises(OracleInputError, match="q @ s"):
        solve_fixed_t([0.5, 0.5], [0.5, 0.5], [1.0, 0.0], 0.2, 1.5)


def test_exact_input_rejects_binary_float() -> None:
    with pytest.raises(OracleInputError, match="floats are not exact inputs"):
        solve_fixed_t_exact(
            q=[0.5, 0.5], p0=["1/2", "1/2"], s=[1, -1], tau="1/5", t="3/2"
        )


def test_exact_fixed_compatibility_threshold_is_rational() -> None:
    threshold = compatibility_threshold_exact(
        q=["1/2", "1/2"], p0=["4/5", "1/5"], tau="1/10"
    )

    assert threshold == Fraction(7, 5)


def test_exact_fixed_handles_mandatory_cap_transport() -> None:
    result = solve_fixed_t_exact(
        q=["1/2", "1/2"],
        p0=["4/5", "1/5"],
        s=[1, -1],
        tau="1/5",
        t="3/2",
    )

    assert result.p == (Fraction(3, 4), Fraction(1, 4))
    assert result.value == Fraction(1, 2)
    assert result.transported_mass == Fraction(1, 20)
    assert result.to_dict()["value"] == "1/2"


def test_exact_integer_serialization_uses_explicit_denominator() -> None:
    result = solve_fixed_t_exact(q=[1], p0=[1], s=[0], tau=0, t=1)

    assert result.to_dict() == {
        "t": "1/1",
        "value": "0/1",
        "p": ["1/1"],
        "transported_mass": "0/1",
    }


def test_exact_fixed_uses_optional_positive_gain_transport() -> None:
    result = solve_fixed_t_exact(
        q=["1/2", "1/2"],
        p0=["1/2", "1/2"],
        s=[1, -1],
        tau="1/5",
        t="3/2",
    )

    assert result.p == (Fraction(7, 10), Fraction(3, 10))
    assert result.value == Fraction(2, 5)
    assert result.transported_mass == Fraction(1, 5)


def test_exact_fixed_rejects_irremovable_zero_support_mass() -> None:
    assert compatibility_threshold_exact(
        q=[1, 0], p0=["7/10", "3/10"], tau="1/5"
    ) is None
    with pytest.raises(IncompatibleAmbiguitySet):
        solve_fixed_t_exact(
            q=[1, 0], p0=["7/10", "3/10"], s=[0, 0], tau="1/5", t=10
        )


def test_exact_crosscheck_matches_scipy_lp_on_rational_cases() -> None:
    rng = np.random.default_rng(20261021)
    for case in range(30):
        n = 3 + case % 2
        q_weights = rng.integers(1, 8, size=n)
        p0_weights = rng.integers(1, 8, size=n)
        q = [Fraction(int(value), int(q_weights.sum())) for value in q_weights]
        p0 = [Fraction(int(value), int(p0_weights.sum())) for value in p0_weights]
        raw_scores = [Fraction(int(value)) for value in rng.integers(-4, 5, size=n)]
        score_mean = sum(
            (q_value * score for q_value, score in zip(q, raw_scores)), Fraction(0)
        )
        s = [score - score_mean for score in raw_scores]
        tau = Fraction(int(rng.integers(1, 5)), 10)
        threshold = compatibility_threshold_exact(q, p0, tau)
        assert threshold is not None
        t = threshold + Fraction(int(rng.integers(1, 6)), 5)

        exact = solve_fixed_t_exact(q, p0, s, tau, t)
        floating = solve_fixed_t(
            [float(value) for value in q],
            [float(value) for value in p0],
            [float(value) for value in s],
            float(tau),
            float(t),
        )

        assert float(exact.value) == pytest.approx(floating.value, abs=1e-8)


def test_exact_outer_respects_restricted_prior_interval() -> None:
    result = solve_mc_tv_pu_oracle_exact(
        q=["1/2", "1/2"],
        p0=["1/2", "1/2"],
        s=[1, -1],
        tau="1/5",
        prior_lower="1/4",
        prior_upper="1/2",
        absolute_gap="1/100000",
    )

    assert result.enclosure_valid
    assert result.certified
    assert result.lower_bound == result.upper_bound == Fraction(4, 5)
    assert result.best_t == Fraction(2)
    assert result.to_dict()["lower_bound"] == "4/5"


def test_exact_infinity_limit_for_negative_tail() -> None:
    result = solve_mc_tv_pu_oracle_exact(
        q=["9/10", "1/10"],
        p0=[1, 0],
        s=[-1, 9],
        tau="1/100",
        prior_lower=0,
        prior_upper="1/2",
        absolute_gap="1/100000",
    )

    assert result.enclosure_valid
    assert result.certified
    assert result.infinity_limit == Fraction(-9, 10)
    assert result.best_t is None
    assert result.lower_bound == result.upper_bound == Fraction(-9, 10)


@pytest.mark.parametrize(
    ("scores", "expected_limit"),
    [([1, -1], Fraction(2, 5)), ([0, 0], Fraction(0))],
)
def test_exact_infinity_records_positive_and_zero_tail_limits(
    scores: list[int], expected_limit: Fraction
) -> None:
    result = solve_mc_tv_pu_oracle_exact(
        q=["1/2", "1/2"],
        p0=["1/2", "1/2"],
        s=scores,
        tau="1/5",
        prior_lower=0,
        prior_upper="1/2",
    )

    assert result.infinity_limit == expected_limit
    assert result.remaining_intervals == 0


def test_exact_zero_support_equal_tau_remains_compatible() -> None:
    q = ["1/2", "1/2", 0]
    p0 = ["2/5", "2/5", "1/5"]

    assert compatibility_threshold_exact(q, p0, "1/5") == 1
    result = solve_fixed_t_exact(q, p0, [1, -1, 0], "1/5", 2)
    assert result.p[2] == 0


def test_exact_threshold_crosses_multiple_ratio_breakpoints() -> None:
    threshold = compatibility_threshold_exact(
        q=["1/5", "1/5", "3/5"],
        p0=["2/5", "3/10", "3/10"],
        tau="1/20",
    )

    assert threshold == Fraction(7, 4)


def test_exact_degenerate_prior_point_interval() -> None:
    result = solve_mc_tv_pu_oracle_exact(
        q=["1/2", "1/2"],
        p0=["1/2", "1/2"],
        s=[1, -1],
        tau="1/5",
        prior_lower="1/2",
        prior_upper="1/2",
    )

    assert result.certified
    assert result.lower_bound == result.upper_bound == Fraction(4, 5)
    assert result.best_t == Fraction(2)
    assert result.to_dict()["diagnostics"]["domain_t"] == ["2/1", "2/1"]
    assert result.to_dict()["diagnostics"]["saturation_t"] == "2/1"


def test_exact_compatible_selector_separates_from_product_relaxation() -> None:
    q = [Fraction(1, 8), Fraction(1, 8), Fraction(1, 4), Fraction(1, 2)]
    p0 = [Fraction(1, 8), Fraction(1, 8), Fraction(3, 8), Fraction(3, 8)]
    loss_a = [Fraction(1, 16), Fraction(3, 16), Fraction(7, 8), Fraction(1, 2)]
    loss_b = [Fraction(1, 16), Fraction(11, 16), Fraction(7, 8), Fraction(3, 8)]
    score_difference = [a - b for a, b in zip(loss_a, loss_b)]

    assert score_difference == [0, Fraction(-1, 2), 0, Fraction(1, 8)]
    assert sum(
        (weight * score for weight, score in zip(q, score_difference)),
        Fraction(0),
    ) == 0

    compatible_a = solve_mc_tv_pu_oracle_exact(
        q=q,
        p0=p0,
        s=score_difference,
        tau=Fraction(1, 5),
        prior_lower=Fraction(1, 5),
        prior_upper=Fraction(3, 4),
        absolute_gap=Fraction(1, 2000),
    )
    compatible_b = solve_mc_tv_pu_oracle_exact(
        q=q,
        p0=p0,
        s=[-value for value in score_difference],
        tau=Fraction(1, 5),
        prior_lower=Fraction(1, 5),
        prior_upper=Fraction(3, 4),
        absolute_gap=Fraction(1, 2000),
    )

    assert compatible_a.lower_bound == compatible_a.upper_bound == Fraction(23, 80)
    assert compatible_b.lower_bound <= Fraction(117, 512) <= compatible_b.upper_bound
    assert compatible_b.upper_bound < compatible_a.lower_bound

    # At t=8 every positive-support cap is at least one, so the fixed-t values
    # equal the two uncoupled TV-ball maxima used by the product relaxation.
    tv_a = solve_fixed_t_exact(q, p0, score_difference, Fraction(1, 5), 8).value
    tv_b = solve_fixed_t_exact(
        q, p0, [-value for value in score_difference], Fraction(1, 5), 8
    ).value
    product_a = 4 * tv_a
    product_b = 4 * tv_b
    assert (product_a, product_b) == (Fraction(23, 80), Fraction(9, 16))
    assert product_a < product_b

    empirical_difference = sum(
        (mass * score for mass, score in zip(p0, score_difference)),
        Fraction(0),
    )
    assert empirical_difference == Fraction(-1, 64)


def test_uniform_q_compatible_selector_separates_from_prior_clipped_rectangle() -> None:
    q = [Fraction(1, 4)] * 4
    p0 = [Fraction(1), Fraction(0), Fraction(0), Fraction(0)]
    loss_a = [Fraction(1, 8), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8)]
    loss_b = [Fraction(3, 8), Fraction(5, 8), Fraction(7, 8), Fraction(1, 8)]
    score_difference = [a - b for a, b in zip(loss_a, loss_b)]
    tau = Fraction(3, 5)

    assert score_difference == [
        Fraction(-1, 4),
        Fraction(-1, 4),
        Fraction(-1, 4),
        Fraction(3, 4),
    ]
    assert sum(
        (weight * score for weight, score in zip(q, score_difference)),
        Fraction(0),
    ) == 0

    geometry = prepare_mc_tv_pu_oracle_exact(q, p0, tau)
    assert geometry.t_min == Fraction(8, 5)
    assert geometry.prior_supremum == Fraction(5, 8)

    compatible_a = solve_mc_tv_pu_oracle_exact(
        q=q,
        p0=p0,
        s=score_difference,
        tau=tau,
        absolute_gap=Fraction(1, 10_000),
    )
    compatible_b = solve_mc_tv_pu_oracle_exact(
        q=q,
        p0=p0,
        s=[-value for value in score_difference],
        tau=tau,
        absolute_gap=Fraction(1, 10_000),
    )
    witness = solve_fixed_t_exact(
        q, p0, score_difference, tau, Fraction(12, 5)
    )
    assert Fraction(12, 5) * witness.value / Fraction(7, 5) == Fraction(3, 5)
    for result in (compatible_a, compatible_b):
        assert result.enclosure_valid is True
        assert result.certified is True
        assert result.upper_bound - result.lower_bound <= Fraction(1, 10_000)
    assert compatible_a.lower_bound <= Fraction(3, 5) <= compatible_a.upper_bound
    assert compatible_a.upper_bound < compatible_b.lower_bound
    assert compatible_b.lower_bound == compatible_b.upper_bound == Fraction(2, 3)

    tv_a = solve_fixed_t_exact(
        q, p0, score_difference, tau, geometry.saturation
    ).value
    tv_b = solve_fixed_t_exact(
        q,
        p0,
        [-value for value in score_difference],
        tau,
        geometry.saturation,
    ).value
    prior_scale = geometry.t_min / (geometry.t_min - 1)
    rectangular_a = prior_scale * tv_a
    rectangular_b = prior_scale * tv_b
    assert (rectangular_a, rectangular_b) == (
        Fraction(14, 15),
        Fraction(2, 3),
    )
    assert rectangular_b < rectangular_a


@pytest.mark.parametrize(
    ("t", "forward_value", "reverse_value"),
    [
        (Fraction(8, 5), Fraction(3, 20), Fraction(1, 4)),
        (Fraction(2), Fraction(1, 4), Fraction(1, 4)),
        (Fraction(12, 5), Fraction(7, 20), Fraction(1, 4)),
        (Fraction(3), Fraction(7, 20), Fraction(1, 4)),
    ],
)
def test_uniform_q_separation_inner_values_match_closed_form(
    t: Fraction,
    forward_value: Fraction,
    reverse_value: Fraction,
) -> None:
    q = [Fraction(1, 4)] * 4
    p0 = [Fraction(1), Fraction(0), Fraction(0), Fraction(0)]
    s = [Fraction(-1, 4)] * 3 + [Fraction(3, 4)]
    tau = Fraction(3, 5)

    assert solve_fixed_t_exact(q, p0, s, tau, t).value == forward_value
    assert solve_fixed_t_exact(
        q, p0, [-value for value in s], tau, t
    ).value == reverse_value


def test_exact_certificate_contains_analytic_interior_peak() -> None:
    result = solve_mc_tv_pu_oracle_exact(
        q=["1/2", "1/2"],
        p0=["4/5", "1/5"],
        s=[1, -1],
        tau="1/10",
        prior_lower="1/4",
        prior_upper=1,
        absolute_gap="1/1000",
        max_evaluations=4096,
    )

    truth = Fraction(9, 5)
    assert result.enclosure_valid
    assert result.certified
    assert result.lower_bound <= truth <= result.upper_bound
    assert result.upper_bound - result.lower_bound <= Fraction(1, 1000)


def test_exact_certificate_budget_exhaustion_keeps_valid_wide_enclosure() -> None:
    result = solve_mc_tv_pu_oracle_exact(
        q=["1/2", "1/2"],
        p0=["4/5", "1/5"],
        s=[1, -1],
        tau="1/10",
        prior_lower="1/4",
        prior_upper=1,
        absolute_gap="1/1000000",
        max_evaluations=2,
    )

    assert result.enclosure_valid
    assert not result.certified
    assert result.lower_bound <= Fraction(9, 5) <= result.upper_bound
    assert result.diagnostics["status"] == "budget_exhausted"


def test_exact_t_one_limit_handles_interior_and_tv_boundary() -> None:
    interior = solve_t_one_limit_exact(
        q=["1/2", "1/2"], p0=["1/2", "1/2"], s=[1, -1], tau="1/5"
    )
    locked = solve_t_one_limit_exact(
        q=["1/2", "1/2"], p0=["7/13", "6/13"], s=["-7/2", "7/2"], tau="1/26"
    )
    decreasing_tv = solve_t_one_limit_exact(
        q=["1/2", "1/2"], p0=["6/13", "7/13"], s=[-1, 1], tau="1/26"
    )

    assert interior.value == 1
    assert interior.regime == "tv_interior"
    assert locked.value == 0
    assert locked.regime == "tv_boundary_dual"
    assert decreasing_tv.value == 1
    assert decreasing_tv.regime == "tv_boundary_dual"


def test_exact_t_one_limit_locks_when_above_center_set_is_empty() -> None:
    directional = solve_t_one_limit_exact(
        q=["1/2", "1/2"], p0=["1/2", "1/2"], s=[1, -1], tau=0
    )
    result = solve_mc_tv_pu_oracle_exact(
        q=["1/2", "1/2"], p0=["1/2", "1/2"], s=[1, -1], tau=0
    )

    assert directional.regime == "tv_boundary_locked"
    assert directional.value == 0
    assert result.left_limit == 0
    assert result.best_t is None
    assert result.lower_bound == result.upper_bound == 0
    assert result.certified


def test_exact_t_one_limit_handles_empty_not_above_set_and_zero_support() -> None:
    directional = solve_t_one_limit_exact(
        q=["1/2", "1/2", 0],
        p0=["2/5", "2/5", "1/5"],
        s=[1, -1, 99],
        tau="1/5",
    )

    assert directional.regime == "tv_boundary_zero_support_only"
    assert directional.value == 1
    assert directional.dual_lambda == 0
    assert directional.dual_mu == -1


def test_exact_t_one_limit_zero_q_coordinate_cannot_change_interior_value() -> None:
    directional = solve_t_one_limit_exact(
        q=["1/2", "1/2", 0],
        p0=["1/2", "1/2", 0],
        s=[1, -1, 999],
        tau="1/5",
    )

    assert directional.regime == "tv_interior"
    assert directional.value == 1
    assert directional.dual_mu == -1


def test_exact_open_boundary_single_support_saturation_is_certified_zero() -> None:
    result = solve_mc_tv_pu_oracle_exact(
        q=[1, 0],
        p0=["4/5", "1/5"],
        s=[0, 3],
        tau="1/5",
        prior_lower=0,
        prior_upper=1,
    )

    assert result.left_limit == 0
    assert result.best_t is None
    assert result.lower_bound == result.upper_bound == 0
    assert result.infinity_limit == 0
    assert result.diagnostics["saturation_t"] == "1/1"
    assert result.diagnostics["t_one_directional"]["regime"] == (
        "tv_boundary_zero_support_only"
    )
    assert result.enclosure_valid
    assert result.certified


def test_exact_t_one_limit_matches_float_lp_on_dyadic_boundary_cases() -> None:
    rng = np.random.default_rng(20261025)
    for _ in range(30):
        q_counts = rng.multinomial(16, np.full(4, 0.25))
        p0_counts = rng.multinomial(16, np.full(4, 0.25))
        q = [Fraction(int(value), 16) for value in q_counts]
        p0 = [Fraction(int(value), 16) for value in p0_counts]
        raw_scores = [Fraction(int(value)) for value in rng.integers(-4, 5, size=4)]
        mean = sum(
            (mass * score for mass, score in zip(q, raw_scores)), Fraction(0)
        )
        scores = [score - mean for score in raw_scores]
        tau = sum(
            (abs(q_value - p0_value) for q_value, p0_value in zip(q, p0)),
            Fraction(0),
        ) / 2

        exact = solve_t_one_limit_exact(q, p0, scores, tau)
        floating = solve_t_one_limit(
            [float(value) for value in q],
            [float(value) for value in p0],
            [float(value) for value in scores],
            float(tau),
        )

        assert float(exact.value) == pytest.approx(floating.value, abs=1e-9)


def test_exact_open_boundary_enclosure_contains_analytic_peak() -> None:
    result = solve_mc_tv_pu_oracle_exact(
        q=["1/2", "1/2"],
        p0=["1/2", "1/2"],
        s=[1, -1],
        tau="1/5",
        prior_lower="1/2",
        prior_upper=1,
        absolute_gap="1/1000",
    )

    assert result.certified
    assert result.left_limit == 1
    assert result.lower_bound <= Fraction(7, 5) <= result.upper_bound
    assert result.upper_bound - result.lower_bound <= Fraction(1, 1000)
    assert result.diagnostics["t_one_directional"]["regime"] == "tv_interior"


def test_compatibility_threshold_matches_two_point_solution() -> None:
    q = np.array([0.5, 0.5])
    p0 = np.array([0.8, 0.2])
    tau = 0.1

    threshold = compatibility_threshold(q, p0, tau)

    assert threshold == pytest.approx(1.4, abs=1e-12)
    assert compatibility_excess(threshold, q, p0) == pytest.approx(tau)


def test_compatibility_detects_irremovable_zero_support_mass() -> None:
    q = np.array([1.0, 0.0])
    p0 = np.array([0.7, 0.3])

    assert math.isinf(compatibility_threshold(q, p0, tau=0.2))
    with pytest.raises(IncompatibleAmbiguitySet):
        solve_fixed_t(q, p0, [0.0, 0.0], tau=0.2, t=10.0)


def test_fixed_t_lp_matches_hand_solved_two_point_problem() -> None:
    q = np.array([0.5, 0.5])
    p0 = np.array([0.8, 0.2])
    s = np.array([1.0, -1.0])

    result = solve_fixed_t(q, p0, s, tau=0.2, t=1.5)

    # TV permits p_1 <= 1.0, while the mixture cap gives p_1 <= 0.75.
    assert result.value == pytest.approx(0.5, abs=1e-10)
    assert result.p == pytest.approx((0.75, 0.25), abs=1e-10)


def test_boundary_limit_with_inactive_tv_constraint() -> None:
    result = solve_t_one_limit(
        q=[0.5, 0.5], p0=[0.5, 0.5], s=[1.0, -1.0], tau=0.2
    )

    assert result.value == pytest.approx(1.0, abs=1e-10)
    assert result.direction == pytest.approx((0.5, -0.5), abs=1e-10)


def test_boundary_limit_respects_active_tv_tangent() -> None:
    result = solve_t_one_limit(
        q=[0.5, 0.5], p0=[0.75, 0.25], s=[-1.0, 1.0], tau=0.25
    )

    assert result.value == pytest.approx(0.0, abs=1e-10)
    assert result.direction == pytest.approx((0.0, 0.0), abs=1e-10)


def test_tail_is_constant_after_saturation_threshold() -> None:
    q = [0.5, 0.5]
    p0 = [0.5, 0.5]
    s = [1.0, -1.0]
    tau = 0.2

    threshold = saturation_threshold(q)
    at_threshold = solve_fixed_t(q, p0, s, tau, threshold)
    far_tail = solve_fixed_t(q, p0, s, tau, 100.0)

    assert threshold == pytest.approx(2.0)
    assert at_threshold.value == pytest.approx(0.4, abs=1e-10)
    assert far_tail.value == pytest.approx(at_threshold.value, abs=1e-10)


def test_global_oracle_matches_analytic_two_point_solution() -> None:
    result = solve_mc_tv_pu_oracle(
        q=[0.5, 0.5],
        p0=[0.5, 0.5],
        s=[1.0, -1.0],
        tau=0.2,
        absolute_gap=1e-6,
    )

    assert not result.certified
    assert result.structural_gap_met
    assert result.lower_bound <= 1.4 <= result.upper_bound
    assert result.upper_bound - result.lower_bound <= 1e-6
    assert result.best_t == pytest.approx(1.4, abs=2e-5)
    assert result.left_limit == pytest.approx(1.0, abs=1e-9)
    assert result.infinity_limit == pytest.approx(0.4, abs=1e-9)


def test_global_oracle_handles_nonunit_compatibility_threshold() -> None:
    result = solve_mc_tv_pu_oracle(
        q=[0.5, 0.5],
        p0=[0.8, 0.2],
        s=[1.0, -1.0],
        tau=0.1,
        absolute_gap=1e-6,
    )

    assert not result.certified
    assert result.structural_gap_met
    assert result.t_min == pytest.approx(1.4, abs=1e-10)
    assert result.prior_supremum == pytest.approx(1.0 / 1.4, abs=1e-10)
    assert result.lower_bound <= 1.8 <= result.upper_bound
    assert result.best_t == pytest.approx(1.8, abs=2e-5)


def test_certificate_reports_budget_or_numeric_limit() -> None:
    result = solve_mc_tv_pu_oracle(
        q=[0.5, 0.5],
        p0=[0.5, 0.5],
        s=[1.0, -1.0],
        tau=0.2,
        absolute_gap=1e-15,
        max_lp_evaluations=2,
        lp_value_tolerance=1e-9,
    )

    assert not result.certified
    assert not result.structural_gap_met
    assert result.upper_bound > result.lower_bound
    assert result.diagnostics["status"] == "budget_exhausted"


def test_global_oracle_respects_restricted_prior_interval() -> None:
    result = solve_mc_tv_pu_oracle(
        q=[0.5, 0.5],
        p0=[0.5, 0.5],
        s=[1.0, -1.0],
        tau=0.2,
        prior_lower=0.25,
        prior_upper=0.5,
        absolute_gap=1e-7,
    )

    assert not result.certified
    assert result.structural_gap_met
    assert result.lower_bound <= 0.8 <= result.upper_bound
    assert result.best_t == pytest.approx(2.0, abs=1e-10)


def test_global_oracle_tail_only_domain_does_not_use_saturation_point() -> None:
    result = solve_mc_tv_pu_oracle(
        q=[0.5, 0.5],
        p0=[0.5, 0.5],
        s=[1.0, -1.0],
        tau=0.2,
        prior_lower=0.1,
        prior_upper=0.2,
        absolute_gap=1e-7,
    )

    assert not result.certified
    assert result.structural_gap_met
    assert result.lower_bound <= 0.5 <= result.upper_bound
    assert result.best_t == pytest.approx(5.0, abs=1e-10)


def test_near_one_compatible_threshold_is_not_treated_as_open_boundary() -> None:
    result = solve_mc_tv_pu_oracle(
        q=[0.5, 0.5],
        p0=[0.501, 0.499],
        s=[1.0, -1.0],
        tau=0.0,
        absolute_gap=1e-5,
    )

    assert result.t_min > 1.0
    assert result.left_limit is None


def test_global_oracle_contains_dense_grid_maximum() -> None:
    q = np.array([0.2, 0.3, 0.5])
    p0 = np.array([0.5, 0.2, 0.3])
    s = np.array([1.0, -0.5, -0.1])
    tau = 0.15
    prior_lower = 0.15
    prior_upper = 0.85

    result = solve_mc_tv_pu_oracle(
        q,
        p0,
        s,
        tau,
        prior_lower=prior_lower,
        prior_upper=prior_upper,
        absolute_gap=2e-6,
    )
    grid_lower = max(result.t_min, 1.0 / prior_upper)
    grid_upper = 1.0 / prior_lower
    grid_values = []
    for t in np.linspace(grid_lower, grid_upper, 121):
        inner = solve_fixed_t(q, p0, s, tau, float(t))
        grid_values.append(t * inner.value / (t - 1.0))
    best_index = int(np.argmax(grid_values))
    left_index = max(0, best_index - 1)
    right_index = min(len(grid_values) - 1, best_index + 1)
    grid = np.linspace(grid_lower, grid_upper, 121)

    def negative_objective(t: float) -> float:
        inner = solve_fixed_t(q, p0, s, tau, t)
        return -(t * inner.value / (t - 1.0))

    refined = minimize_scalar(
        negative_objective,
        bounds=(float(grid[left_index]), float(grid[right_index])),
        method="bounded",
        options={"xatol": 1e-12},
    )
    independent_maximum = max(max(grid_values), -float(refined.fun))

    assert not result.certified
    assert result.structural_gap_met
    assert result.lower_bound - 2e-6 <= independent_maximum <= result.upper_bound + 2e-6
    assert result.upper_bound - result.lower_bound <= 2e-6


def test_degenerate_single_support_has_exact_zero_objective() -> None:
    result = solve_mc_tv_pu_oracle(
        q=[1.0, 0.0], p0=[0.8, 0.2], s=[0.0, 3.0], tau=0.2
    )

    assert result.certified
    assert result.structural_gap_met
    assert result.lower_bound == result.upper_bound == 0.0
    assert result.lp_evaluations == 0


def test_global_oracle_is_deterministic_and_serializable() -> None:
    kwargs = {
        "q": [0.5, 0.5],
        "p0": [0.5, 0.5],
        "s": [1.0, -1.0],
        "tau": 0.2,
        "absolute_gap": 1e-6,
    }

    first = solve_mc_tv_pu_oracle(**kwargs)
    second = solve_mc_tv_pu_oracle(**kwargs)

    assert first.to_dict() == second.to_dict()


def test_near_tv_boundary_interior_is_not_collapsed_to_tangent_boundary() -> None:
    result = solve_mc_tv_pu_oracle(
        q=[0.5, 0.5],
        p0=[0.69999999995, 0.30000000005],
        s=[-1.0, 1.0],
        tau=0.2,
        absolute_gap=0.01,
    )

    true_supremum = 1.0000000001
    assert result.left_limit == pytest.approx(1.0, abs=1e-9)
    assert result.lower_bound <= true_supremum <= result.upper_bound


def test_zero_support_mass_strictly_above_tau_is_incompatible() -> None:
    with pytest.raises(IncompatibleAmbiguitySet):
        solve_mc_tv_pu_oracle(
            q=[1.0, 0.0],
            p0=[0.8, 0.2],
            s=[0.0, 3.0],
            tau=0.2 - 5e-11,
        )


def test_strictly_empty_prior_intersection_is_rejected() -> None:
    incompatible_prior = 0.7142857142867142
    with pytest.raises(IncompatibleAmbiguitySet):
        solve_mc_tv_pu_oracle(
            q=[0.5, 0.5],
            p0=[0.8, 0.2],
            s=[1.0, -1.0],
            tau=0.1,
            prior_lower=incompatible_prior,
            prior_upper=incompatible_prior,
        )


def test_tail_only_best_t_remains_inside_requested_domain() -> None:
    domain_lower = 2.00000000005
    result = solve_mc_tv_pu_oracle(
        q=[0.5, 0.5],
        p0=[0.5, 0.5],
        s=[1.0, -1.0],
        tau=0.2,
        prior_lower=0.4,
        prior_upper=1.0 / domain_lower,
        absolute_gap=1e-7,
    )

    assert result.best_t is not None
    assert domain_lower <= result.best_t <= 2.5
    assert result.best_t == pytest.approx(domain_lower, abs=1e-12)


def test_tiny_positive_support_is_not_treated_as_zero() -> None:
    epsilon = 5e-11
    q = [1.0 - epsilon, epsilon]
    s = [-1.0, (1.0 - epsilon) / epsilon]

    result = solve_t_one_limit(q=q, p0=q, s=s, tau=0.1)

    assert result.value == pytest.approx(1.0, abs=1e-8)
    assert result.direction == pytest.approx((-epsilon, epsilon), abs=1e-12)


def test_negative_tail_uses_infinity_limit_without_finite_witness() -> None:
    result = solve_mc_tv_pu_oracle(
        q=[0.9, 0.1],
        p0=[1.0, 0.0],
        s=[-1.0, 9.0],
        tau=0.01,
        prior_lower=0.0,
        prior_upper=0.5,
        absolute_gap=1e-6,
    )

    assert result.infinity_limit == pytest.approx(-0.9, abs=1e-9)
    assert result.best_t is None
    assert result.lower_bound <= -0.9 <= result.upper_bound


@pytest.mark.parametrize("residual", [5e-10, -5e-10])
def test_single_support_recenters_accepted_centering_residual(residual: float) -> None:
    result = solve_mc_tv_pu_oracle(
        q=[1.0, 0.0],
        p0=[0.8, 0.2],
        s=[residual, 3.0],
        tau=0.2,
    )

    assert result.certified
    assert result.lower_bound == result.upper_bound == 0.0
    assert result.diagnostics["input_centering_residual"] == pytest.approx(residual)
    assert result.diagnostics["recentered"] is True


def test_exact_tv_boundary_keeps_subtolerance_nonzero_signs() -> None:
    delta = 2.0**-34
    q = [0.25, 0.25, 0.5]
    p0 = [0.25 - delta, 0.25 + delta, 0.5]

    directional = solve_t_one_limit(q=q, p0=p0, s=[1.0, -1.0, 0.0], tau=delta)
    result = solve_mc_tv_pu_oracle(
        q=q,
        p0=p0,
        s=[1.0, -1.0, 0.0],
        tau=delta,
        absolute_gap=1e-5,
    )

    assert directional.value == pytest.approx(0.0, abs=1e-12)
    assert result.left_limit == pytest.approx(0.0, abs=1e-12)
    assert result.best_value == pytest.approx(0.0, abs=1e-8)


def test_single_support_near_one_prior_uses_finite_domain_witness() -> None:
    prior_upper = 1.0 - 5e-11
    result = solve_mc_tv_pu_oracle(
        q=[1.0, 0.0],
        p0=[0.8, 0.2],
        s=[0.0, 3.0],
        tau=0.2,
        prior_lower=0.5,
        prior_upper=prior_upper,
    )

    assert result.left_limit is None
    assert result.best_t == pytest.approx(1.0 / prior_upper, abs=1e-14)
