"""Numerical oracle for PU-compatible total-variation ambiguity sets.

The module implements the finite-support optimization specified in
``implement-stage/mc-tv-pu-oracle/SPEC.md``.  It deliberately accepts an
explicit target measure ``q`` and never infers its scientific meaning from a
dataset.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any, TypeAlias

import numpy as np
from scipy.optimize import linprog


class OracleInputError(ValueError):
    """Raised when inputs do not define the specified finite-support problem."""


class IncompatibleAmbiguitySet(ValueError):
    """Raised when no legal prior in the requested interval is compatible."""


class LinearProgramError(RuntimeError):
    """Raised when HiGHS does not return an optimal LP solution."""


RationalInput: TypeAlias = Fraction | int | str


def _fraction_string(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True)
class FixedTResult:
    """Optimal value and witness for one finite value of ``t``."""

    t: float
    value: float
    p: tuple[float, ...]
    supergradient: float


@dataclass(frozen=True)
class DirectionalLimitResult:
    """Right-hand derivative of ``G`` at ``t=1`` and a direction witness."""

    value: float
    direction: tuple[float, ...]


@dataclass(frozen=True)
class ExactFixedTResult:
    """Exact fixed-``t`` optimum over rational inputs."""

    t: Fraction
    value: Fraction
    p: tuple[Fraction, ...]
    transported_mass: Fraction

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": _fraction_string(self.t),
            "value": _fraction_string(self.value),
            "p": [_fraction_string(value) for value in self.p],
            "transported_mass": _fraction_string(self.transported_mass),
        }


@dataclass(frozen=True)
class _ExactFixedTValueResult:
    """Internal fixed-``t`` value without an unused probability witness."""

    t: Fraction
    value: Fraction


@dataclass(frozen=True)
class ExactDirectionalLimitResult:
    """Exact right-hand limit data at the open boundary ``t=1``."""

    value: Fraction
    dual_lambda: Fraction
    dual_mu: Fraction
    regime: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": _fraction_string(self.value),
            "dual_lambda": _fraction_string(self.dual_lambda),
            "dual_mu": _fraction_string(self.dual_mu),
            "regime": self.regime,
        }


@dataclass(frozen=True)
class ExactOracleResult:
    """Rigorous rational enclosure for the supported exact outer problem."""

    lower_bound: Fraction
    upper_bound: Fraction
    best_t: Fraction | None
    best_value: Fraction
    t_min: Fraction
    prior_supremum: Fraction
    evaluations: int
    remaining_intervals: int
    enclosure_valid: bool
    certified: bool
    absolute_gap: Fraction
    left_limit: Fraction | None
    infinity_limit: Fraction | None
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower_bound": _fraction_string(self.lower_bound),
            "upper_bound": _fraction_string(self.upper_bound),
            "best_t": None if self.best_t is None else _fraction_string(self.best_t),
            "best_value": _fraction_string(self.best_value),
            "t_min": _fraction_string(self.t_min),
            "prior_supremum": _fraction_string(self.prior_supremum),
            "evaluations": self.evaluations,
            "remaining_intervals": self.remaining_intervals,
            "enclosure_valid": self.enclosure_valid,
            "certified": self.certified,
            "absolute_gap": _fraction_string(self.absolute_gap),
            "left_limit": (
                None if self.left_limit is None else _fraction_string(self.left_limit)
            ),
            "infinity_limit": (
                None
                if self.infinity_limit is None
                else _fraction_string(self.infinity_limit)
            ),
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class ExactOracleGeometry:
    """Validated score-independent geometry for repeated exact oracle calls."""

    q: tuple[Fraction, ...]
    p0: tuple[Fraction, ...]
    tau: Fraction
    prior_lower: Fraction
    prior_upper: Fraction
    t_min: Fraction
    prior_supremum: Fraction
    domain_lower: Fraction
    domain_upper: Fraction | None
    saturation: Fraction


@dataclass(frozen=True)
class OracleResult:
    """Conditional numerical enclosure for the outer supremum.

    Only results with ``certified=True`` are treated as certificates.  Ordinary
    floating-point LP paths currently report ``structural_gap_met`` while
    keeping ``certified=False``.
    """

    lower_bound: float
    upper_bound: float
    best_t: float | None
    best_value: float
    t_min: float
    prior_supremum: float
    lp_evaluations: int
    remaining_intervals: int
    certified: bool
    structural_gap_met: bool
    absolute_gap: float
    left_limit: float | None
    infinity_limit: float | None
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _as_probability_vector(
    name: str,
    values: np.ndarray | list[float] | tuple[float, ...],
    *,
    probability_tolerance: float,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise OracleInputError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise OracleInputError(f"{name} must contain only finite values")
    if np.min(array) < -probability_tolerance:
        raise OracleInputError(f"{name} must be non-negative")
    array = np.maximum(array, 0.0)
    total = float(np.sum(array))
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=probability_tolerance):
        raise OracleInputError(f"{name} must sum to one; observed {total:.17g}")
    return array / total


def _to_fraction(name: str, value: RationalInput) -> Fraction:
    if isinstance(value, (bool, float)):
        raise OracleInputError(
            f"{name} must use Fraction, int, or a rational string; floats are not exact inputs"
        )
    if isinstance(value, Fraction):
        return value
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, str):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError) as error:
            raise OracleInputError(f"{name} is not a valid rational string") from error
    raise OracleInputError(
        f"{name} must use Fraction, int, or a rational string"
    )


def _as_exact_vector(
    name: str,
    values: list[RationalInput] | tuple[RationalInput, ...],
) -> tuple[Fraction, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise OracleInputError(f"{name} must be a non-empty list or tuple")
    return tuple(_to_fraction(f"{name}[{index}]", value) for index, value in enumerate(values))


def _validated_exact_inputs(
    q: list[RationalInput] | tuple[RationalInput, ...],
    p0: list[RationalInput] | tuple[RationalInput, ...],
    s: list[RationalInput] | tuple[RationalInput, ...],
    tau: RationalInput,
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...], Fraction]:
    q_exact = _as_exact_vector("q", q)
    p0_exact = _as_exact_vector("p0", p0)
    s_exact = _as_exact_vector("s", s)
    tau_exact = _to_fraction("tau", tau)
    if not (len(q_exact) == len(p0_exact) == len(s_exact)):
        raise OracleInputError("q, p0, and s must have the same length")
    if any(value < 0 for value in q_exact) or sum(q_exact) != 1:
        raise OracleInputError("exact q must be non-negative and sum exactly to one")
    if any(value < 0 for value in p0_exact) or sum(p0_exact) != 1:
        raise OracleInputError("exact p0 must be non-negative and sum exactly to one")
    if not 0 <= tau_exact <= 1:
        raise OracleInputError("exact tau must lie in [0, 1]")
    if sum(q_value * s_value for q_value, s_value in zip(q_exact, s_exact)) != 0:
        raise OracleInputError("exact inputs require q @ s to equal zero exactly")
    return q_exact, p0_exact, s_exact, tau_exact


def compatibility_excess_exact(
    t: RationalInput,
    q: list[RationalInput] | tuple[RationalInput, ...],
    p0: list[RationalInput] | tuple[RationalInput, ...],
) -> Fraction:
    """Return the exact compatibility excess for rational inputs."""

    t_exact = _to_fraction("t", t)
    q_exact = _as_exact_vector("q", q)
    p0_exact = _as_exact_vector("p0", p0)
    if len(q_exact) != len(p0_exact):
        raise OracleInputError("q and p0 must have the same length")
    if t_exact < 1:
        raise OracleInputError("exact t must be at least one")
    return sum(
        (max(p0_value - t_exact * q_value, Fraction(0)) for q_value, p0_value in zip(q_exact, p0_exact)),
        Fraction(0),
    )


def compatibility_threshold_exact(
    q: list[RationalInput] | tuple[RationalInput, ...],
    p0: list[RationalInput] | tuple[RationalInput, ...],
    tau: RationalInput,
) -> Fraction | None:
    """Return the exact smallest compatible ``t``, or ``None`` if impossible."""

    q_exact = _as_exact_vector("q", q)
    p0_exact = _as_exact_vector("p0", p0)
    tau_exact = _to_fraction("tau", tau)
    if len(q_exact) != len(p0_exact):
        raise OracleInputError("q and p0 must have the same length")
    if any(value < 0 for value in q_exact) or sum(q_exact) != 1:
        raise OracleInputError("exact q must be non-negative and sum exactly to one")
    if any(value < 0 for value in p0_exact) or sum(p0_exact) != 1:
        raise OracleInputError("exact p0 must be non-negative and sum exactly to one")
    if not 0 <= tau_exact <= 1:
        raise OracleInputError("exact tau must lie in [0, 1]")
    if compatibility_excess_exact(1, q_exact, p0_exact) <= tau_exact:
        return Fraction(1)

    zero_support_mass = sum(
        (p0_value for q_value, p0_value in zip(q_exact, p0_exact) if q_value == 0),
        Fraction(0),
    )
    if zero_support_mass > tau_exact:
        return None

    ratios = sorted(
        {
            p0_value / q_value
            for q_value, p0_value in zip(q_exact, p0_exact)
            if q_value > 0 and p0_value / q_value > 1
        }
    )
    lower = Fraction(1)
    for upper in ratios:
        active = [
            index
            for index, (q_value, p0_value) in enumerate(zip(q_exact, p0_exact))
            if q_value > 0 and p0_value / q_value > lower
        ]
        intercept = zero_support_mass + sum(
            (p0_exact[index] for index in active), Fraction(0)
        )
        slope = sum((q_exact[index] for index in active), Fraction(0))
        if slope > 0:
            root = (intercept - tau_exact) / slope
            if lower <= root <= upper:
                return root
        lower = upper
    if compatibility_excess_exact(lower, q_exact, p0_exact) <= tau_exact:
        return lower
    raise AssertionError("exact compatibility threshold construction failed")


def solve_fixed_t_exact(
    q: list[RationalInput] | tuple[RationalInput, ...],
    p0: list[RationalInput] | tuple[RationalInput, ...],
    s: list[RationalInput] | tuple[RationalInput, ...],
    tau: RationalInput,
    t: RationalInput,
) -> ExactFixedTResult:
    """Solve the fixed-``t`` problem exactly by rational mass transport."""

    q_exact, p0_exact, s_exact, tau_exact = _validated_exact_inputs(q, p0, s, tau)
    t_exact = _to_fraction("t", t)
    if t_exact < 1:
        raise OracleInputError("exact t must be at least one")
    threshold = compatibility_threshold_exact(q_exact, p0_exact, tau_exact)
    if threshold is None or t_exact < threshold:
        raise IncompatibleAmbiguitySet(
            f"exact t={t_exact} is below the compatibility threshold {threshold}"
        )

    return _solve_fixed_t_exact_prevalidated(
        q_exact, p0_exact, s_exact, tau_exact, t_exact
    )


def _solve_fixed_t_exact_prevalidated(
    q_exact: tuple[Fraction, ...],
    p0_exact: tuple[Fraction, ...],
    s_exact: tuple[Fraction, ...],
    tau_exact: Fraction,
    t_exact: Fraction,
    score_orders: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
) -> ExactFixedTResult:
    """Exact mass transport after compatibility and vector validation."""

    n = len(q_exact)
    caps = tuple(t_exact * value for value in q_exact)
    mandatory_removal = [
        max(p0_exact[index] - caps[index], Fraction(0)) for index in range(n)
    ]
    mandatory_mass = sum(mandatory_removal, Fraction(0))
    p = [p0_exact[index] - mandatory_removal[index] for index in range(n)]
    recipient_capacity = [
        max(caps[index] - p0_exact[index], Fraction(0)) for index in range(n)
    ]

    recipient_order, donor_order = (
        _score_orders(s_exact) if score_orders is None else score_orders
    )
    remaining_mandatory = mandatory_mass
    for index in recipient_order:
        amount = min(recipient_capacity[index], remaining_mandatory)
        p[index] += amount
        recipient_capacity[index] -= amount
        remaining_mandatory -= amount
        if remaining_mandatory == 0:
            break
    if remaining_mandatory != 0:
        raise AssertionError("compatible cap lacks redistribution capacity")

    donor_capacity = [
        p0_exact[index] - mandatory_removal[index] for index in range(n)
    ]
    budget = tau_exact - mandatory_mass
    donor_pointer = 0
    recipient_pointer = 0
    while budget > 0:
        while donor_pointer < n and donor_capacity[donor_order[donor_pointer]] == 0:
            donor_pointer += 1
        while recipient_pointer < n and recipient_capacity[recipient_order[recipient_pointer]] == 0:
            recipient_pointer += 1
        if donor_pointer == n or recipient_pointer == n:
            break
        donor = donor_order[donor_pointer]
        recipient = recipient_order[recipient_pointer]
        if s_exact[recipient] <= s_exact[donor]:
            break
        amount = min(donor_capacity[donor], recipient_capacity[recipient], budget)
        p[donor] -= amount
        p[recipient] += amount
        donor_capacity[donor] -= amount
        recipient_capacity[recipient] -= amount
        budget -= amount

    transported_mass = sum(
        (abs(p[index] - p0_exact[index]) for index in range(n)), Fraction(0)
    ) / 2
    if sum(p, Fraction(0)) != 1:
        raise AssertionError("exact transport did not preserve probability mass")
    if any(value < 0 or value > caps[index] for index, value in enumerate(p)):
        raise AssertionError("exact transport violated a probability or cap constraint")
    if transported_mass > tau_exact:
        raise AssertionError("exact transport exceeded the TV budget")
    value = sum((s_exact[index] * p[index] for index in range(n)), Fraction(0))
    return ExactFixedTResult(
        t=t_exact,
        value=value,
        p=tuple(p),
        transported_mass=transported_mass,
    )


def _solve_fixed_t_exact_value_prevalidated(
    q_exact: tuple[Fraction, ...],
    p0_exact: tuple[Fraction, ...],
    s_exact: tuple[Fraction, ...],
    tau_exact: Fraction,
    t_exact: Fraction,
    score_orders: tuple[tuple[int, ...], tuple[int, ...]],
    baseline_value: Fraction,
) -> _ExactFixedTValueResult:
    """Evaluate the exact transport objective without materializing its witness."""

    n = len(q_exact)
    caps = tuple(t_exact * value for value in q_exact)
    mandatory_removal = [
        max(p0_exact[index] - caps[index], Fraction(0)) for index in range(n)
    ]
    mandatory_mass = sum(mandatory_removal, Fraction(0))
    recipient_capacity = [
        max(caps[index] - p0_exact[index], Fraction(0)) for index in range(n)
    ]
    value = baseline_value - sum(
        (
            s_exact[index] * mandatory_removal[index]
            for index in range(n)
        ),
        Fraction(0),
    )

    recipient_order, donor_order = score_orders
    remaining_mandatory = mandatory_mass
    for index in recipient_order:
        amount = min(recipient_capacity[index], remaining_mandatory)
        value += s_exact[index] * amount
        recipient_capacity[index] -= amount
        remaining_mandatory -= amount
        if remaining_mandatory == 0:
            break
    if remaining_mandatory != 0:
        raise AssertionError("compatible cap lacks redistribution capacity")

    donor_capacity = [
        p0_exact[index] - mandatory_removal[index] for index in range(n)
    ]
    budget = tau_exact - mandatory_mass
    donor_pointer = 0
    recipient_pointer = 0
    while budget > 0:
        while donor_pointer < n and donor_capacity[donor_order[donor_pointer]] == 0:
            donor_pointer += 1
        while recipient_pointer < n and recipient_capacity[recipient_order[recipient_pointer]] == 0:
            recipient_pointer += 1
        if donor_pointer == n or recipient_pointer == n:
            break
        donor = donor_order[donor_pointer]
        recipient = recipient_order[recipient_pointer]
        if s_exact[recipient] <= s_exact[donor]:
            break
        amount = min(donor_capacity[donor], recipient_capacity[recipient], budget)
        value += (s_exact[recipient] - s_exact[donor]) * amount
        donor_capacity[donor] -= amount
        recipient_capacity[recipient] -= amount
        budget -= amount

    return _ExactFixedTValueResult(t=t_exact, value=value)


@lru_cache(maxsize=256)
def _score_orders(
    s_exact: tuple[Fraction, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return stable recipient and donor orders for one fixed score direction."""

    indices = range(len(s_exact))
    recipient_order = tuple(
        sorted(indices, key=lambda index: (-s_exact[index], index))
    )
    donor_order = tuple(
        sorted(range(len(s_exact)), key=lambda index: (s_exact[index], index))
    )
    return recipient_order, donor_order


def solve_t_one_limit_exact(
    q: list[RationalInput] | tuple[RationalInput, ...],
    p0: list[RationalInput] | tuple[RationalInput, ...],
    s: list[RationalInput] | tuple[RationalInput, ...],
    tau: RationalInput,
) -> ExactDirectionalLimitResult:
    """Compute the exact ``t -> 1+`` limit from the TV tangent-cone dual."""

    q_exact, p0_exact, s_exact, tau_exact = _validated_exact_inputs(q, p0, s, tau)
    return _solve_t_one_limit_exact_prevalidated(q_exact, p0_exact, s_exact, tau_exact)


def _solve_t_one_limit_exact_prevalidated(
    q_exact: tuple[Fraction, ...],
    p0_exact: tuple[Fraction, ...],
    s_exact: tuple[Fraction, ...],
    tau_exact: Fraction,
) -> ExactDirectionalLimitResult:
    """Exact open-boundary limit after geometry and score validation."""

    tv_at_q = sum(
        (abs(q_value - p0_value) for q_value, p0_value in zip(q_exact, p0_exact)),
        Fraction(0),
    ) / 2
    if tv_at_q > tau_exact:
        raise IncompatibleAmbiguitySet(
            "the exact t -> 1+ limit is unavailable because q lies outside the TV ball"
        )

    support = [index for index, value in enumerate(q_exact) if value > 0]
    minimum_score = min(s_exact[index] for index in support)
    if tv_at_q < tau_exact:
        return ExactDirectionalLimitResult(
            value=-minimum_score,
            dual_lambda=Fraction(0),
            dual_mu=minimum_score,
            regime="tv_interior",
        )

    above_center = [
        index for index in support if q_exact[index] > p0_exact[index]
    ]
    not_above_center = [
        index for index in support if q_exact[index] <= p0_exact[index]
    ]
    if not above_center:
        return ExactDirectionalLimitResult(
            value=Fraction(0),
            dual_lambda=Fraction(0),
            dual_mu=minimum_score,
            regime="tv_boundary_locked",
        )
    if not not_above_center:
        return ExactDirectionalLimitResult(
            value=-minimum_score,
            dual_lambda=Fraction(0),
            dual_mu=minimum_score,
            regime="tv_boundary_zero_support_only",
        )

    a = min(s_exact[index] for index in above_center)
    b = min(s_exact[index] for index in not_above_center)
    dual_lambda = max(Fraction(0), (a - b) / 2)
    dual_mu = min(a - dual_lambda, b + dual_lambda)
    value = sum(
        (
            q_exact[index] * (s_exact[index] - dual_mu - dual_lambda)
            for index in above_center
        ),
        Fraction(0),
    )
    value += sum(
        (
            q_exact[index] * (s_exact[index] - dual_mu + dual_lambda)
            for index in support
            if q_exact[index] < p0_exact[index]
        ),
        Fraction(0),
    )
    value += sum(
        (
            q_exact[index]
            * max(Fraction(0), s_exact[index] - dual_mu - dual_lambda)
            for index in support
            if q_exact[index] == p0_exact[index]
        ),
        Fraction(0),
    )
    if value < 0:
        raise AssertionError("exact tangent-cone dual returned a negative support value")
    return ExactDirectionalLimitResult(
        value=value,
        dual_lambda=dual_lambda,
        dual_mu=dual_mu,
        regime="tv_boundary_dual",
    )


def prepare_mc_tv_pu_oracle_exact(
    q: list[RationalInput] | tuple[RationalInput, ...],
    p0: list[RationalInput] | tuple[RationalInput, ...],
    tau: RationalInput,
    prior_lower: RationalInput = 0,
    prior_upper: RationalInput = 1,
) -> ExactOracleGeometry:
    """Validate and cache the score-independent exact ambiguity geometry."""

    q_exact = _as_exact_vector("q", q)
    p0_exact = _as_exact_vector("p0", p0)
    tau_exact = _to_fraction("tau", tau)
    if len(q_exact) != len(p0_exact):
        raise OracleInputError("q and p0 must have the same length")
    if any(value < 0 for value in q_exact) or sum(q_exact) != 1:
        raise OracleInputError("exact q must be non-negative and sum exactly to one")
    if any(value < 0 for value in p0_exact) or sum(p0_exact) != 1:
        raise OracleInputError("exact p0 must be non-negative and sum exactly to one")
    if not 0 <= tau_exact <= 1:
        raise OracleInputError("exact tau must lie in [0, 1]")
    prior_lower_exact = _to_fraction("prior_lower", prior_lower)
    prior_upper_exact = _to_fraction("prior_upper", prior_upper)
    if not 0 <= prior_lower_exact <= prior_upper_exact <= 1:
        raise OracleInputError(
            "exact prior bounds must satisfy 0 <= lower <= upper <= 1"
        )
    if prior_upper_exact == 0 or prior_lower_exact == 1:
        raise IncompatibleAmbiguitySet(
            "the exact prior interval contains no value in (0, 1)"
        )
    t_min = compatibility_threshold_exact(q_exact, p0_exact, tau_exact)
    if t_min is None:
        raise IncompatibleAmbiguitySet(
            "the exact TV ball cannot remove enough zero-support mass"
        )
    prior_supremum = min(prior_upper_exact, Fraction(1, 1) / t_min)
    domain_lower = max(t_min, Fraction(1, 1) / prior_upper_exact)
    domain_upper = (
        None
        if prior_lower_exact == 0
        else Fraction(1, 1) / prior_lower_exact
    )
    if domain_upper is not None and domain_lower > domain_upper:
        raise IncompatibleAmbiguitySet(
            "the exact compatible and requested prior intervals do not intersect"
        )
    positive_q = [value for value in q_exact if value > 0]
    saturation = Fraction(1, 1) / min(positive_q)
    return ExactOracleGeometry(
        q=q_exact,
        p0=p0_exact,
        tau=tau_exact,
        prior_lower=prior_lower_exact,
        prior_upper=prior_upper_exact,
        t_min=t_min,
        prior_supremum=prior_supremum,
        domain_lower=domain_lower,
        domain_upper=domain_upper,
        saturation=saturation,
    )


def solve_mc_tv_pu_oracle_exact(
    q: list[RationalInput] | tuple[RationalInput, ...],
    p0: list[RationalInput] | tuple[RationalInput, ...],
    s: list[RationalInput] | tuple[RationalInput, ...],
    tau: RationalInput,
    prior_lower: RationalInput = 0,
    prior_upper: RationalInput = 1,
    *,
    absolute_gap: RationalInput = "1/1000",
    max_evaluations: int = 4096,
) -> ExactOracleResult:
    """Return a rigorous rational enclosure using exact branch-and-bound."""

    geometry = prepare_mc_tv_pu_oracle_exact(
        q, p0, tau, prior_lower=prior_lower, prior_upper=prior_upper
    )
    return solve_mc_tv_pu_oracle_exact_prepared(
        geometry,
        s,
        absolute_gap=absolute_gap,
        max_evaluations=max_evaluations,
    )


def solve_mc_tv_pu_oracle_exact_prepared(
    geometry: ExactOracleGeometry,
    s: list[RationalInput] | tuple[RationalInput, ...],
    *,
    absolute_gap: RationalInput = "1/1000",
    max_evaluations: int = 4096,
) -> ExactOracleResult:
    """Solve repeated score directions against one validated exact geometry."""

    if not isinstance(geometry, ExactOracleGeometry):
        raise OracleInputError("geometry must be prepared by prepare_mc_tv_pu_oracle_exact")
    s_exact = _as_exact_vector("s", s)
    if len(s_exact) != len(geometry.q):
        raise OracleInputError("prepared geometry and s must have the same length")
    if sum(
        (q_value * s_value for q_value, s_value in zip(geometry.q, s_exact)),
        Fraction(0),
    ) != 0:
        raise OracleInputError("exact inputs require q @ s to equal zero exactly")
    gap_exact = _to_fraction("absolute_gap", absolute_gap)
    if gap_exact <= 0:
        raise OracleInputError("exact absolute_gap must be positive")
    if type(max_evaluations) is not int or max_evaluations < 2:
        raise OracleInputError("max_evaluations must be an integer of at least two")

    q_exact = geometry.q
    p0_exact = geometry.p0
    tau_exact = geometry.tau
    t_min = geometry.t_min
    prior_supremum = geometry.prior_supremum
    domain_lower = geometry.domain_lower
    domain_upper = geometry.domain_upper
    saturation = geometry.saturation
    cache: dict[Fraction, _ExactFixedTValueResult] = {}
    score_orders = _score_orders(s_exact)
    baseline_value = sum(
        (
            s_value * p0_value
            for s_value, p0_value in zip(s_exact, p0_exact, strict=True)
        ),
        Fraction(0),
    )

    def evaluate(t_value: Fraction) -> _ExactFixedTValueResult:
        if t_value not in cache:
            if len(cache) >= max_evaluations:
                raise RuntimeError("exact evaluation budget exhausted")
            cache[t_value] = _solve_fixed_t_exact_value_prevalidated(
                q_exact,
                p0_exact,
                s_exact,
                tau_exact,
                t_value,
                score_orders,
                baseline_value,
            )
        return cache[t_value]

    def transformed(t_value: Fraction, g_value: Fraction) -> Fraction:
        return t_value * g_value / (t_value - 1)

    directional = (
        _solve_t_one_limit_exact_prevalidated(q_exact, p0_exact, s_exact, tau_exact)
        if domain_lower == 1
        else None
    )
    left_limit = None if directional is None else directional.value
    best_lower: Fraction | None = left_limit
    best_value: Fraction | None = left_limit
    best_t: Fraction | None = None

    def add_candidate(result: _ExactFixedTValueResult) -> None:
        nonlocal best_lower, best_value, best_t
        value = transformed(result.t, result.value)
        if best_lower is None or value > best_lower:
            best_lower = value
        if best_value is None or value > best_value:
            best_value = value
            best_t = result.t

    pre_upper = saturation if domain_upper is None else min(domain_upper, saturation)
    start_result: _ExactFixedTValueResult | None = None
    end_result: _ExactFixedTValueResult | None = None
    if domain_lower > 1 and domain_lower <= pre_upper:
        start_result = evaluate(domain_lower)
        add_candidate(start_result)
        end_result = evaluate(pre_upper)
        add_candidate(end_result)
    elif domain_lower == 1 and pre_upper > 1:
        end_result = evaluate(pre_upper)
        add_candidate(end_result)

    intervals: list[
        tuple[
            Fraction,
            int,
            Fraction,
            Fraction,
            _ExactFixedTValueResult | None,
            _ExactFixedTValueResult,
        ]
    ] = []
    counter = 0

    def interval_upper(
        left: Fraction, right: Fraction, right_result: _ExactFixedTValueResult
    ) -> Fraction:
        if left == 1:
            if directional is None:
                raise AssertionError("open t=1 interval lacks a directional limit")
            return right * directional.value
        if right_result.value >= 0:
            return transformed(left, right_result.value)
        return transformed(right, right_result.value)

    def push_interval(
        left: Fraction,
        right: Fraction,
        left_result: _ExactFixedTValueResult | None,
        right_result: _ExactFixedTValueResult,
    ) -> None:
        nonlocal counter
        if right <= left:
            return
        upper = interval_upper(left, right, right_result)
        counter += 1
        heapq.heappush(
            intervals,
            (-upper, counter, left, right, left_result, right_result),
        )

    if (
        start_result is not None
        and end_result is not None
        and pre_upper > domain_lower
    ):
        push_interval(
            domain_lower, pre_upper, start_result, end_result
        )
    elif domain_lower == 1 and end_result is not None:
        push_interval(domain_lower, pre_upper, None, end_result)

    terminal_upper: Fraction | None = None
    infinity_limit: Fraction | None = None
    tail_present = domain_upper is None or domain_upper > saturation
    if tail_present:
        tail_start = max(domain_lower, saturation)
        tail_result = evaluate(tail_start)
        if domain_upper is None:
            infinity_limit = tail_result.value
        if tail_start == 1:
            if tail_result.value != 0:
                raise AssertionError("a t=1 saturated tail must have zero centered value")
            tail_optimum = Fraction(0)
            tail_best_t = None
        elif tail_result.value >= 0:
            tail_optimum = transformed(tail_start, tail_result.value)
            tail_best_t: Fraction | None = tail_start
        elif domain_upper is None:
            tail_optimum = tail_result.value
            tail_best_t = None
        else:
            tail_optimum = transformed(domain_upper, tail_result.value)
            tail_best_t = domain_upper
        terminal_upper = tail_optimum
        if best_lower is None or tail_optimum > best_lower:
            best_lower = tail_optimum
        if best_value is None or tail_optimum > best_value:
            best_value = tail_optimum
            best_t = tail_best_t

    if best_lower is None or best_value is None:
        raise AssertionError("exact domain produced no candidate or tail limit")

    status = "gap_met"
    while True:
        heap_upper = -intervals[0][0] if intervals else None
        upper_candidates = [best_lower]
        if heap_upper is not None:
            upper_candidates.append(heap_upper)
        if terminal_upper is not None:
            upper_candidates.append(terminal_upper)
        global_upper = max(upper_candidates)
        if global_upper - best_lower <= gap_exact:
            break
        if not intervals or len(cache) >= max_evaluations:
            status = "budget_exhausted"
            break

        _, _, left, right, left_result, right_result = heapq.heappop(intervals)
        midpoint = (left + right) / 2
        midpoint_result = evaluate(midpoint)
        add_candidate(midpoint_result)
        push_interval(left, midpoint, left_result, midpoint_result)
        push_interval(midpoint, right, midpoint_result, right_result)

    heap_upper = -intervals[0][0] if intervals else None
    final_upper_candidates = [best_lower]
    if heap_upper is not None:
        final_upper_candidates.append(heap_upper)
    if terminal_upper is not None:
        final_upper_candidates.append(terminal_upper)
    global_upper = max(final_upper_candidates)
    certified = global_upper - best_lower <= gap_exact
    if certified:
        status = "gap_met"

    return ExactOracleResult(
        lower_bound=best_lower,
        upper_bound=global_upper,
        best_t=best_t,
        best_value=best_value,
        t_min=t_min,
        prior_supremum=prior_supremum,
        evaluations=len(cache),
        remaining_intervals=len(intervals),
        enclosure_valid=True,
        certified=certified,
        absolute_gap=gap_exact,
        left_limit=left_limit,
        infinity_limit=infinity_limit,
        diagnostics={
            "status": status,
            "domain_t": [
                _fraction_string(domain_lower),
                None if domain_upper is None else _fraction_string(domain_upper),
            ],
            "saturation_t": _fraction_string(saturation),
            "arithmetic": "fractions.Fraction",
            "certificate_type": "exact_monotone_branch_and_bound",
            "t_one_directional": (
                None if directional is None else directional.to_dict()
            ),
        },
    )


def _validated_inputs(
    q: np.ndarray | list[float] | tuple[float, ...],
    p0: np.ndarray | list[float] | tuple[float, ...],
    s: np.ndarray | list[float] | tuple[float, ...],
    tau: float,
    *,
    probability_tolerance: float,
    centering_tolerance: float,
    loss_bound: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if not math.isfinite(tau) or not 0.0 <= tau <= 1.0:
        raise OracleInputError("tau must be finite and lie in [0, 1]")
    if probability_tolerance <= 0.0 or centering_tolerance <= 0.0:
        raise OracleInputError("validation tolerances must be positive")

    q_array = _as_probability_vector(
        "q", q, probability_tolerance=probability_tolerance
    )
    p0_array = _as_probability_vector(
        "p0", p0, probability_tolerance=probability_tolerance
    )
    s_array = np.asarray(s, dtype=np.float64)
    if s_array.ndim != 1 or s_array.shape != q_array.shape:
        raise OracleInputError("s must be one-dimensional and match q and p0")
    if not np.all(np.isfinite(s_array)):
        raise OracleInputError("s must contain only finite values")
    centered = float(q_array @ s_array)
    if abs(centered) > centering_tolerance:
        raise OracleInputError(
            "q @ s must be zero for the symmetric pairwise-loss construction; "
            f"observed {centered:.17g}"
        )
    s_array = s_array - centered
    if loss_bound is not None:
        if not math.isfinite(loss_bound) or loss_bound <= 0.0:
            raise OracleInputError("loss_bound must be positive when provided")
        if float(np.max(np.abs(s_array))) > loss_bound + centering_tolerance:
            raise OracleInputError("abs(s) exceeds the declared loss_bound")
    return q_array, p0_array, s_array, centered


def compatibility_excess(
    t: float,
    q: np.ndarray | list[float] | tuple[float, ...],
    p0: np.ndarray | list[float] | tuple[float, ...],
) -> float:
    """Return ``sum_i max(p0_i - t q_i, 0)`` without normalizing inputs."""

    if not math.isfinite(t) or t < 1.0:
        raise OracleInputError("t must be finite and at least one")
    q_array = np.asarray(q, dtype=np.float64)
    p0_array = np.asarray(p0, dtype=np.float64)
    if q_array.shape != p0_array.shape or q_array.ndim != 1:
        raise OracleInputError("q and p0 must be matching one-dimensional arrays")
    return float(np.maximum(p0_array - t * q_array, 0.0).sum())


def compatibility_threshold(
    q: np.ndarray | list[float] | tuple[float, ...],
    p0: np.ndarray | list[float] | tuple[float, ...],
    tau: float,
    *,
    probability_tolerance: float = 1e-12,
) -> float:
    """Return the smallest finite ``t >= 1`` satisfying compatibility.

    ``math.inf`` is returned when mass on the zero support of ``q`` exceeds the
    TV radius, so no finite mixture proportion is compatible.
    """

    if not math.isfinite(tau) or not 0.0 <= tau <= 1.0:
        raise OracleInputError("tau must be finite and lie in [0, 1]")
    q_array = _as_probability_vector(
        "q", q, probability_tolerance=probability_tolerance
    )
    p0_array = _as_probability_vector(
        "p0", p0, probability_tolerance=probability_tolerance
    )
    if compatibility_excess(1.0, q_array, p0_array) <= tau:
        return 1.0

    positive = q_array > 0.0
    zero_support_mass = float(np.sum(p0_array[~positive]))
    if zero_support_mass > tau:
        return math.inf

    ratios = p0_array[positive] / q_array[positive]
    high = max(1.0, float(np.max(ratios)))
    low = 1.0
    for _ in range(200):
        midpoint = (low + high) / 2.0
        if compatibility_excess(midpoint, q_array, p0_array) <= tau:
            high = midpoint
        else:
            low = midpoint
    return high


def solve_fixed_t(
    q: np.ndarray | list[float] | tuple[float, ...],
    p0: np.ndarray | list[float] | tuple[float, ...],
    s: np.ndarray | list[float] | tuple[float, ...],
    tau: float,
    t: float,
    *,
    probability_tolerance: float = 1e-10,
    centering_tolerance: float = 1e-9,
    loss_bound: float | None = None,
) -> FixedTResult:
    """Solve the inner TV-and-mixture-cap LP for a fixed finite ``t``."""

    q_array, p0_array, s_array, _ = _validated_inputs(
        q,
        p0,
        s,
        tau,
        probability_tolerance=probability_tolerance,
        centering_tolerance=centering_tolerance,
        loss_bound=loss_bound,
    )
    if not math.isfinite(t) or t < 1.0:
        raise OracleInputError("t must be finite and at least one")
    threshold = compatibility_threshold(
        q_array, p0_array, tau, probability_tolerance=probability_tolerance
    )
    if not math.isfinite(threshold) or t < threshold:
        raise IncompatibleAmbiguitySet(
            f"t={t:.17g} is below the compatibility threshold {threshold:.17g}"
        )

    n = q_array.size
    objective = np.concatenate((-s_array, np.zeros(n, dtype=np.float64)))

    cap = np.zeros((n, 2 * n), dtype=np.float64)
    cap[np.arange(n), np.arange(n)] = 1.0

    upper_abs = np.zeros((n, 2 * n), dtype=np.float64)
    upper_abs[np.arange(n), np.arange(n)] = 1.0
    upper_abs[np.arange(n), n + np.arange(n)] = -1.0

    lower_abs = np.zeros((n, 2 * n), dtype=np.float64)
    lower_abs[np.arange(n), np.arange(n)] = -1.0
    lower_abs[np.arange(n), n + np.arange(n)] = -1.0

    tv_row = np.zeros((1, 2 * n), dtype=np.float64)
    tv_row[0, n:] = 1.0

    a_ub = np.vstack((cap, upper_abs, lower_abs, tv_row))
    b_ub = np.concatenate(
        (t * q_array, p0_array, -p0_array, np.array([2.0 * tau]))
    )
    a_eq = np.zeros((1, 2 * n), dtype=np.float64)
    a_eq[0, :n] = 1.0

    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * (2 * n),
        method="highs",
    )
    if not result.success:
        raise LinearProgramError(
            f"fixed-t LP failed with status {result.status}: {result.message}"
        )
    p = np.asarray(result.x[:n], dtype=np.float64)
    value = float(s_array @ p)
    cap_marginals = np.asarray(result.ineqlin.marginals[:n], dtype=np.float64)
    supergradient = max(0.0, float(-(q_array @ cap_marginals)))
    return FixedTResult(
        t=float(t),
        value=value,
        p=tuple(float(x) for x in p),
        supergradient=supergradient,
    )


def saturation_threshold(
    q: np.ndarray | list[float] | tuple[float, ...],
    *,
    probability_tolerance: float = 1e-12,
) -> float:
    """Return a finite ``t`` after which all positive-support caps are inert."""

    q_array = _as_probability_vector(
        "q", q, probability_tolerance=probability_tolerance
    )
    positive = q_array[q_array > 0.0]
    return float(1.0 / np.min(positive))


def solve_t_one_limit(
    q: np.ndarray | list[float] | tuple[float, ...],
    p0: np.ndarray | list[float] | tuple[float, ...],
    s: np.ndarray | list[float] | tuple[float, ...],
    tau: float,
    *,
    probability_tolerance: float = 1e-10,
    centering_tolerance: float = 1e-9,
    loss_bound: float | None = None,
) -> DirectionalLimitResult:
    """Compute ``lim_{t -> 1+} t G(t)/(t-1)`` by a tangent LP.

    The limit exists only when ``q`` itself lies in the TV ball.  At a TV
    boundary the directional derivative of the L1 norm is enforced explicitly,
    including absolute-value auxiliaries for coordinates where ``q == p0``.
    """

    q_array, p0_array, s_array, _ = _validated_inputs(
        q,
        p0,
        s,
        tau,
        probability_tolerance=probability_tolerance,
        centering_tolerance=centering_tolerance,
        loss_bound=loss_bound,
    )
    tv_at_q = 0.5 * float(np.sum(np.abs(q_array - p0_array)))
    if tv_at_q > tau:
        raise IncompatibleAmbiguitySet(
            "the t -> 1+ limit is unavailable because q lies outside the TV ball"
        )

    n = q_array.size
    on_tv_boundary = tv_at_q == tau
    zero_difference = q_array - p0_array == 0.0
    zero_indices = np.flatnonzero(zero_difference) if on_tv_boundary else np.array([], dtype=int)
    auxiliary_count = int(zero_indices.size)
    variable_count = n + auxiliary_count

    objective = np.zeros(variable_count, dtype=np.float64)
    objective[:n] = -s_array
    a_eq = np.zeros((1, variable_count), dtype=np.float64)
    a_eq[0, :n] = 1.0
    a_ub_rows: list[np.ndarray] = []
    b_ub: list[float] = []

    if on_tv_boundary:
        tangent = np.zeros(variable_count, dtype=np.float64)
        difference = q_array - p0_array
        tangent[:n][difference > 0.0] = 1.0
        tangent[:n][difference < 0.0] = -1.0
        for offset, index in enumerate(zero_indices):
            auxiliary_index = n + offset
            tangent[auxiliary_index] = 1.0

            positive_abs = np.zeros(variable_count, dtype=np.float64)
            positive_abs[index] = 1.0
            positive_abs[auxiliary_index] = -1.0
            a_ub_rows.append(positive_abs)
            b_ub.append(0.0)

            negative_abs = np.zeros(variable_count, dtype=np.float64)
            negative_abs[index] = -1.0
            negative_abs[auxiliary_index] = -1.0
            a_ub_rows.append(negative_abs)
            b_ub.append(0.0)

        a_ub_rows.append(tangent)
        b_ub.append(0.0)

    bounds: list[tuple[float | None, float | None]] = []
    for q_value in q_array:
        if q_value == 0.0:
            bounds.append((0.0, 0.0))
        else:
            bounds.append((None, float(q_value)))
    bounds.extend([(0.0, None)] * auxiliary_count)

    result = linprog(
        objective,
        A_ub=np.vstack(a_ub_rows) if a_ub_rows else None,
        b_ub=np.asarray(b_ub, dtype=np.float64) if b_ub else None,
        A_eq=a_eq,
        b_eq=np.array([0.0]),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise LinearProgramError(
            f"t=1 directional LP failed with status {result.status}: {result.message}"
        )
    direction = np.asarray(result.x[:n], dtype=np.float64)
    value = max(0.0, float(s_array @ direction))
    return DirectionalLimitResult(
        value=value, direction=tuple(float(x) for x in direction)
    )


def solve_mc_tv_pu_oracle(
    q: np.ndarray | list[float] | tuple[float, ...],
    p0: np.ndarray | list[float] | tuple[float, ...],
    s: np.ndarray | list[float] | tuple[float, ...],
    tau: float,
    prior_lower: float = 0.0,
    prior_upper: float = 1.0,
    *,
    absolute_gap: float = 1e-6,
    max_lp_evaluations: int = 4096,
    probability_tolerance: float = 1e-10,
    centering_tolerance: float = 1e-9,
    lp_value_tolerance: float = 1e-9,
    loss_bound: float | None = None,
) -> OracleResult:
    """Bound the full compatible-prior supremum by deterministic branch-and-bound.

    Each interval combines two valid upper bounds for the concave, nondecreasing
    inner value ``G``: monotonicity and the HiGHS dual supergradient at the left
    endpoint.  The latter gives a supporting affine majorant whose transformed
    objective can attain a maximum only at an interval endpoint.
    """

    q_array, p0_array, s_array, input_centering_residual = _validated_inputs(
        q,
        p0,
        s,
        tau,
        probability_tolerance=probability_tolerance,
        centering_tolerance=centering_tolerance,
        loss_bound=loss_bound,
    )
    if (
        not math.isfinite(prior_lower)
        or not math.isfinite(prior_upper)
        or not 0.0 <= prior_lower <= prior_upper <= 1.0
    ):
        raise OracleInputError(
            "prior bounds must be finite and satisfy 0 <= lower <= upper <= 1"
        )
    if prior_upper <= 0.0 or prior_lower >= 1.0:
        raise IncompatibleAmbiguitySet(
            "the requested prior interval contains no value in (0, 1)"
        )
    if not math.isfinite(absolute_gap) or absolute_gap <= 0.0:
        raise OracleInputError("absolute_gap must be finite and positive")
    if max_lp_evaluations < 2:
        raise OracleInputError("max_lp_evaluations must be at least two")
    if not math.isfinite(lp_value_tolerance) or lp_value_tolerance <= 0.0:
        raise OracleInputError("lp_value_tolerance must be finite and positive")

    t_min = compatibility_threshold(
        q_array, p0_array, tau, probability_tolerance=probability_tolerance
    )
    prior_supremum = (
        0.0
        if not math.isfinite(t_min)
        else min(prior_upper, 1.0 / t_min)
    )
    if not math.isfinite(t_min):
        raise IncompatibleAmbiguitySet(
            "the TV ball cannot remove enough mass outside the support of q"
        )

    domain_lower = max(t_min, 1.0 / prior_upper)
    domain_upper = math.inf if prior_lower == 0.0 else 1.0 / prior_lower
    if domain_lower > domain_upper or domain_upper <= 1.0:
        raise IncompatibleAmbiguitySet(
            "no compatible prior lies in the requested interval; "
            f"t_min={t_min:.17g}, prior=[{prior_lower:.17g}, {prior_upper:.17g}]"
        )

    saturation = saturation_threshold(
        q_array, probability_tolerance=probability_tolerance
    )
    finite_upper = min(domain_upper, saturation)
    scale = max(1.0, float(np.max(np.abs(s_array))))
    value_slack = lp_value_tolerance * scale
    slope_slack = lp_value_tolerance * max(
        1.0, float(np.max(s_array) - np.min(s_array))
    )

    if saturation <= 1.0 + probability_tolerance:
        left_available = domain_lower <= 1.0
        infinity_available = math.isinf(domain_upper)
        return OracleResult(
            lower_bound=0.0,
            upper_bound=0.0,
            best_t=None if left_available else float(domain_lower),
            best_value=0.0,
            t_min=float(t_min),
            prior_supremum=float(prior_supremum),
            lp_evaluations=0,
            remaining_intervals=0,
            certified=True,
            structural_gap_met=True,
            absolute_gap=float(absolute_gap),
            left_limit=0.0 if left_available else None,
            infinity_limit=0.0 if infinity_available else None,
            diagnostics={
                "status": "degenerate_single_support",
                "domain_t": [
                    float(domain_lower),
                    None if infinity_available else float(domain_upper),
                ],
                "saturation_t": float(saturation),
                "value_slack": float(value_slack),
                "slope_slack": float(slope_slack),
                "certificate_type": "analytic_degenerate_support",
                "input_centering_residual": float(input_centering_residual),
                "recentered": True,
            },
        )

    cache: dict[float, FixedTResult] = {}

    def evaluate(t: float) -> FixedTResult:
        key = float(t)
        if key not in cache:
            if len(cache) >= max_lp_evaluations:
                raise RuntimeError("LP evaluation budget exhausted")
            cache[key] = solve_fixed_t(
                q_array,
                p0_array,
                s_array,
                tau,
                key,
                probability_tolerance=probability_tolerance,
                centering_tolerance=centering_tolerance,
                loss_bound=loss_bound,
            )
        return cache[key]

    def weight(t: float) -> float:
        return t / (t - 1.0)

    best_lower = -math.inf
    best_raw = -math.inf
    best_t: float | None = None

    def add_finite_candidate(result: FixedTResult) -> None:
        nonlocal best_lower, best_raw, best_t
        raw = weight(result.t) * result.value
        lower = weight(result.t) * (result.value - value_slack)
        best_lower = max(best_lower, lower)
        if raw > best_raw:
            best_raw = raw
            best_t = result.t

    left_limit: float | None = None
    infinity_limit: float | None = None
    terminal_upper_bounds: list[float] = []
    left_open = domain_lower <= 1.0
    start = domain_lower

    if left_open:
        directional = solve_t_one_limit(
            q_array,
            p0_array,
            s_array,
            tau,
            probability_tolerance=probability_tolerance,
            centering_tolerance=centering_tolerance,
            loss_bound=loss_bound,
        )
        left_limit = directional.value
        left_lower = directional.value - value_slack
        best_lower = max(best_lower, left_lower)
        if directional.value > best_raw:
            best_raw = directional.value
            best_t = None

        delta = absolute_gap / (8.0 * max(directional.value + value_slack, 1.0))
        proposed_start = max(float(np.nextafter(1.0, math.inf)), 1.0 + delta)
        start = min(finite_upper, proposed_start)
        terminal_upper_bounds.append(
            (directional.value + value_slack) * start
        )

    pre_saturation_present = finite_upper >= start
    if finite_upper > 1.0 and pre_saturation_present:
        start_result = evaluate(start)
        add_finite_candidate(start_result)
        end_result = evaluate(finite_upper)
        add_finite_candidate(end_result)
    else:
        start_result = None
        end_result = None

    counter = 0
    intervals: list[
        tuple[float, int, float, float, FixedTResult, FixedTResult]
    ] = []

    def interval_upper(
        left: float,
        right: float,
        left_result: FixedTResult,
        right_result: FixedTResult,
    ) -> float:
        right_value_upper = right_result.value + value_slack
        if right_value_upper >= 0.0:
            monotone_upper = weight(left) * right_value_upper
        else:
            monotone_upper = weight(right) * right_value_upper

        left_value_upper = left_result.value + value_slack
        slope_upper = left_result.supergradient + slope_slack
        line_at_right = left_value_upper + slope_upper * (right - left)
        support_upper = max(
            weight(left) * left_value_upper,
            weight(right) * line_at_right,
        )
        return min(monotone_upper, support_upper)

    def push_interval(
        left: float,
        right: float,
        left_result: FixedTResult,
        right_result: FixedTResult,
    ) -> None:
        nonlocal counter
        if right <= left:
            return
        upper = max(
            best_lower,
            interval_upper(left, right, left_result, right_result),
        )
        counter += 1
        heapq.heappush(
            intervals,
            (-upper, counter, left, right, left_result, right_result),
        )

    if (
        start_result is not None
        and end_result is not None
        and finite_upper > start
    ):
        push_interval(start, finite_upper, start_result, end_result)
    elif start_result is not None:
        terminal_upper_bounds.append(
            weight(start_result.t) * (start_result.value + value_slack)
        )

    tail_present = domain_upper > saturation and domain_lower < domain_upper
    if tail_present:
        tail_start = max(domain_lower, saturation)
        tail_result = evaluate(max(tail_start, finite_upper))
        tail_value_lower = tail_result.value - value_slack
        tail_value_upper = tail_result.value + value_slack
        tail_end_weight = 1.0 if math.isinf(domain_upper) else weight(domain_upper)
        if tail_value_upper >= 0.0:
            tail_upper = weight(tail_start) * tail_value_upper
        else:
            tail_upper = tail_end_weight * tail_value_upper
        terminal_upper_bounds.append(tail_upper)

        if tail_result.value >= 0.0:
            raw_tail = weight(tail_start) * tail_result.value
            lower_tail = weight(tail_start) * tail_value_lower
            raw_tail_t: float | None = tail_start
        else:
            raw_tail = tail_end_weight * tail_result.value
            lower_tail = tail_end_weight * tail_value_lower
            raw_tail_t = None if math.isinf(domain_upper) else domain_upper
        best_lower = max(best_lower, lower_tail)
        if raw_tail > best_raw:
            best_raw = raw_tail
            best_t = raw_tail_t
        if math.isinf(domain_upper):
            infinity_limit = tail_result.value

    status = "gap_met"
    while True:
        heap_upper = -intervals[0][0] if intervals else -math.inf
        terminal_upper = max(terminal_upper_bounds, default=-math.inf)
        global_upper = max(best_lower, heap_upper, terminal_upper)
        if global_upper - best_lower <= absolute_gap:
            break
        if not intervals or len(cache) >= max_lp_evaluations:
            status = "budget_exhausted"
            break

        _, _, left, right, left_result, right_result = heapq.heappop(intervals)
        midpoint = (left + right) / 2.0
        if midpoint <= left or midpoint >= right:
            terminal_upper_bounds.append(global_upper)
            status = "floating_point_resolution"
            break
        midpoint_result = evaluate(midpoint)
        add_finite_candidate(midpoint_result)
        push_interval(left, midpoint, left_result, midpoint_result)
        push_interval(midpoint, right, midpoint_result, right_result)

    heap_upper = -intervals[0][0] if intervals else -math.inf
    global_upper = max(
        best_lower,
        heap_upper,
        max(terminal_upper_bounds, default=-math.inf),
    )
    structural_gap_met = global_upper - best_lower <= absolute_gap
    if structural_gap_met:
        status = "structural_gap_met_unverified_lp_error"

    if loss_bound is not None and best_raw > loss_bound + 10.0 * value_slack:
        raise LinearProgramError(
            "computed objective exceeds the declared theoretical loss bound"
        )

    return OracleResult(
        lower_bound=float(best_lower),
        upper_bound=float(global_upper),
        best_t=best_t,
        best_value=float(best_raw),
        t_min=float(t_min),
        prior_supremum=float(prior_supremum),
        lp_evaluations=len(cache),
        remaining_intervals=len(intervals) + len(terminal_upper_bounds),
        certified=False,
        structural_gap_met=structural_gap_met,
        absolute_gap=float(absolute_gap),
        left_limit=left_limit,
        infinity_limit=infinity_limit,
        diagnostics={
            "status": status,
            "domain_t": [
                float(domain_lower),
                None if math.isinf(domain_upper) else float(domain_upper),
            ],
            "saturation_t": float(saturation),
            "value_slack": float(value_slack),
            "slope_slack": float(slope_slack),
            "certificate_type": "conditional_float_enclosure_unverified_lp_error",
            "input_centering_residual": float(input_centering_residual),
            "recentered": True,
        },
    )
