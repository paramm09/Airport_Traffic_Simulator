"""Analytical CPA (Closest Point of Approach) conflict detection.

This is the main DAA algorithm of the project. Given two airborne aircraft
flying at constant velocity, it answers: *"will these two aircraft come closer
than the separation thresholds within the look-ahead horizon?"*

The model is deliberately the classic straight-line kinematic one::

    pA(t) = pA0 + vA * t          d0  = pA0 - pB0
    pB(t) = pB0 + vB * t          rv  = vA - vB
    zA(t) = zA0 + vzA * t         dz0 = zA0 - zB0
    zB(t) = zB0 + vzB * t         rvz = vzA - vzB

    d(t)  = d0  + rv  * t         (horizontal separation)
    dz(t) = dz0 + rvz * t         (vertical separation)

A conflict is a violation of BOTH separation minima at the same instant::

    |d(t)|  <= S_h      and      |dz(t)| <= S_v      for some t in [0, H]

Method (O(1) per pair, no time sampling)
----------------------------------------
``|d(t)|² <= S_h²`` expands into a quadratic inequality in ``t``::

    a t² + b t + c <= 0     with a = rv·rv, b = 2 d0·rv, c = d0·d0 - S_h²

The set of ``t`` satisfying it is an interval (or empty), intersected with
``[0, H]``. The vertical condition is a pair of linear inequalities which also
yields an interval. The conflict exists exactly when the two intervals overlap,
and the earliest point of that overlap is the *time to conflict*.

This is a pure function: it never mutates an aircraft, never decides what to do,
and never owns the response. Resolution is Phase 2F's job, keeping detection
and resolution separable.

Complexity
----------
    detect_conflict         O(1) time, O(1) space
    detect_all_conflicts    O(A^2) time, O(C) space for the C returned results

The ``A^2`` all-pairs scan is deliberate for a project this size; a spatial
index is noted as future work rather than implemented.

NOTE: this is an educational simplification. The separation values are made-up
teaching thresholds, not real aviation separation standards, and constant-velocity
motion is a first-order approximation with no manoeuvring or wind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from backend.models.aircraft import Aircraft
from backend.simulation.airspace import (
    AIRBORNE_STATES,
    velocity_from_heading,
    vertical_rate,
)

# Look-ahead horizon in simulation ticks.
HORIZON = 10.0

# Horizontal separation minimum (distance units).
HORIZONTAL_SEPARATION = 5.0

# Vertical separation minimum (altitude units).
VERTICAL_SEPARATION = 100.0

# Tolerance used for the "is this quadratic term zero?" and interval-overlap
# comparisons, so that floating-point noise does not create or destroy a
# conflict exactly on the threshold.
EPSILON = 1e-9


@dataclass
class ConflictResult:
    """The outcome of one :func:`detect_conflict` call.

    The conflict is symmetric, so ``aircraft_a`` is simply the first argument
    and ``aircraft_b`` the second; swapping the arguments swaps the fields but
    leaves every number unchanged.

    Attributes:
        aircraft_a: The first aircraft (the argument, not "the culprit").
        aircraft_b: The second aircraft.
        conflict: ``True`` when the separation thresholds are violated
            together at some instant inside the horizon.
        time_to_conflict: Earliest time in ``[0, H]`` at which the violation
            occurs, or ``None`` when there is no conflict. This is the start of
            the overlap interval, which is *not* the same as the horizontal CPA
            time.
        horizontal_distance: ``|d(t)|`` at ``time_to_conflict``.
        vertical_distance: ``|dz(t)|`` at ``time_to_conflict``.
        predicted_position_a: Where ``aircraft_a`` is expected to be at
            ``time_to_conflict``.
        predicted_position_b: Same for ``aircraft_b``.
        horizontal_cpa_time: Time of the horizontal closest point of approach,
            clamped into ``[0, H]``; reported even when there is no conflict
            because it is useful for the console. ``None`` when the two
            aircraft have no horizontal relative velocity.
    """

    aircraft_a: Aircraft
    aircraft_b: Aircraft
    conflict: bool
    time_to_conflict: float | None = None
    horizontal_distance: float | None = None
    vertical_distance: float | None = None
    predicted_position_a: tuple[float, float] | None = None
    predicted_position_b: tuple[float, float] | None = None
    horizontal_cpa_time: float | None = None


def _horizontal_interval(
    d0x: float,
    d0y: float,
    rvx: float,
    rvy: float,
    horizon: float,
    separation: float,
) -> tuple[float, float] | None:
    """Return the ``t`` interval where ``|d0 + rv*t| <= separation``, else None.

    Solves the quadratic inequality ``a t^2 + b t + c <= 0`` in closed form:
    when ``a`` is non-zero the answer is the interval between the two roots,
    when ``a`` is zero the inequality is linear and the answer is a half-line.
    Either way the result is intersected with ``[0, horizon]``.
    """
    a = rvx * rvx + rvy * rvy
    b = 2.0 * (d0x * rvx + d0y * rvy)
    c = d0x * d0x + d0y * d0y - separation * separation

    if a < EPSILON:
        # No meaningful horizontal relative motion: the quadratic term
        # vanishes and `b t + c <= 0` is a linear inequality in t.
        if b == 0.0:
            # The separation never changes.
            if c <= 0.0:
                return (0.0, horizon)
            return None
        bound = -c / b
        if b > 0.0:
            # t <= bound: the pair is separating, so only an already-violating
            # pair can conflict, and only before `bound`.
            if bound < 0.0:
                return None
            return (0.0, min(horizon, bound))
        # b < 0: t >= bound, the pair is closing, so the interval runs to the
        # end of the horizon.
        if bound > horizon:
            return None
        return (max(0.0, bound), horizon)

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None

    root = math.sqrt(discriminant)
    low = (-b - root) / (2.0 * a)
    high = (-b + root) / (2.0 * a)
    low = max(low, 0.0)
    high = min(high, horizon)
    if low > high:
        return None
    return (low, high)


def _vertical_interval(
    dz0: float, rvz: float, horizon: float, separation: float
) -> tuple[float, float] | None:
    """Return the ``t`` interval where ``|dz0 + rvz*t| <= separation``, else None.

    Two linear inequalities ``-S_v <= dz0 + rvz t <= S_v``; with no vertical
    relative rate the gap is constant, otherwise the entry and exit instants
    bound the interval.
    """
    if abs(rvz) < EPSILON:
        if abs(dz0) <= separation:
            return (0.0, horizon)
        return None

    # -separation <= dz0 + rvz t <= separation
    t1 = (-separation - dz0) / rvz
    t2 = (separation - dz0) / rvz
    low = max(min(t1, t2), 0.0)
    high = min(max(t1, t2), horizon)
    if low > high:
        return None
    return (low, high)


def _horizontal_cpa_time(
    d0x: float, d0y: float, rvx: float, rvy: float, horizon: float
) -> float | None:
    """Return the clamped time at which the horizontal distance is smallest."""
    speed_squared = rvx * rvx + rvy * rvy
    if speed_squared < EPSILON:
        return None
    t = -(d0x * rvx + d0y * rvy) / speed_squared
    return min(max(t, 0.0), horizon)


def detect_conflict(
    aircraft_a: Aircraft,
    aircraft_b: Aircraft,
    horizon: float = HORIZON,
    horizontal_separation: float = HORIZONTAL_SEPARATION,
    vertical_separation: float = VERTICAL_SEPARATION,
) -> ConflictResult:
    """Detect a loss of separation between two airborne aircraft.

    Args:
        aircraft_a: First aircraft. Its ``position`` and ``heading``/``speed``
            give the horizontal velocity; its vertical rate comes from
            :func:`~backend.simulation.airspace.vertical_rate`.
        aircraft_b: Second aircraft.
        horizon: Look-ahead window in simulation ticks.
        horizontal_separation: Horizontal minimum separation (distance units).
        vertical_separation: Vertical minimum separation (altitude units).

    Returns:
        A :class:`ConflictResult`. It is a *pure* result: neither aircraft is
        modified and no resolution is applied.

    Raises:
        ValueError: if ``horizon`` is negative, a separation is negative, or an
            aircraft is missing.
    """
    if aircraft_a is None or aircraft_b is None:
        raise ValueError("both aircraft are required")
    if horizon < 0.0:
        raise ValueError(f"horizon must not be negative, got {horizon!r}")
    if horizontal_separation < 0.0 or vertical_separation < 0.0:
        raise ValueError("separation thresholds must not be negative")

    # Velocities from the shared heading convention (0 = North, 90 = East).
    vax, vay = velocity_from_heading(aircraft_a.speed, aircraft_a.heading)
    vbx, vby = velocity_from_heading(aircraft_b.speed, aircraft_b.heading)

    d0x = aircraft_a.position[0] - aircraft_b.position[0]
    d0y = aircraft_a.position[1] - aircraft_b.position[1]
    rvx = vax - vbx
    rvy = vay - vby

    dz0 = aircraft_a.altitude - aircraft_b.altitude
    rvz = vertical_rate(aircraft_a) - vertical_rate(aircraft_b)

    cpa_time = _horizontal_cpa_time(d0x, d0y, rvx, rvy, horizon)

    horizontal = _horizontal_interval(
        d0x, d0y, rvx, rvy, horizon, horizontal_separation
    )
    if horizontal is None:
        return ConflictResult(
            aircraft_a=aircraft_a,
            aircraft_b=aircraft_b,
            conflict=False,
            horizontal_cpa_time=cpa_time,
        )

    vertical = _vertical_interval(dz0, rvz, horizon, vertical_separation)
    if vertical is None:
        return ConflictResult(
            aircraft_a=aircraft_a,
            aircraft_b=aircraft_b,
            conflict=False,
            horizontal_cpa_time=cpa_time,
        )

    # A conflict needs both minima violated at the SAME instant, so the two
    # intervals must overlap.
    start = max(horizontal[0], vertical[0])
    end = min(horizontal[1], vertical[1])
    if start > end + EPSILON:
        return ConflictResult(
            aircraft_a=aircraft_a,
            aircraft_b=aircraft_b,
            conflict=False,
            horizontal_cpa_time=cpa_time,
        )

    return ConflictResult(
        aircraft_a=aircraft_a,
        aircraft_b=aircraft_b,
        conflict=True,
        time_to_conflict=start,
        horizontal_distance=math.hypot(d0x + rvx * start, d0y + rvy * start),
        vertical_distance=abs(dz0 + rvz * start),
        predicted_position_a=(
            aircraft_a.position[0] + vax * start,
            aircraft_a.position[1] + vay * start,
        ),
        predicted_position_b=(
            aircraft_b.position[0] + vbx * start,
            aircraft_b.position[1] + vby * start,
        ),
        horizontal_cpa_time=cpa_time,
    )


def detect_all_conflicts(
    aircraft: list[Aircraft],
    horizon: float = HORIZON,
    horizontal_separation: float = HORIZONTAL_SEPARATION,
    vertical_separation: float = VERTICAL_SEPARATION,
) -> list[ConflictResult]:
    """Check every unordered pair of airborne aircraft.

    Each pair is examined exactly once, so ``n`` airborne aircraft produce
    ``n(n-1)/2`` results. Ground aircraft (not airborne) are ignored.

    Returns:
        Every pair result, including the pairs with no conflict, so that a
        caller (or a test) can see the whole picture. The list order is
        deterministic: ``(a, b)`` for ``a`` before ``b`` in the input list.
    """
    airborne = [
        item for item in aircraft if item.state in AIRBORNE_STATES
    ]

    results: list[ConflictResult] = []
    for first in range(len(airborne)):
        for second in range(first + 1, len(airborne)):
            results.append(
                detect_conflict(
                    airborne[first],
                    airborne[second],
                    horizon=horizon,
                    horizontal_separation=horizontal_separation,
                    vertical_separation=vertical_separation,
                )
            )
    return results
