"""Phase 2E tests: analytical CPA conflict detection.

The production detector is closed-form. The tests contain an INDEPENDENT
time-sampling brute-force oracle that simply walks the horizon in small steps
and checks the separation thresholds directly. Where the two agree over many
random-free scenarios, the quadratic algebra is validated without using the
same maths twice.
"""

import math
import pytest

from backend.models.aircraft import Aircraft, AircraftState
from backend.models.waypoint import Waypoint
from backend.simulation.airspace import (
    AIRBORNE_STATES,
    ALTITUDE_RATE,
    velocity_from_heading,
    vertical_rate,
)
from backend.simulation.conflict import (
    EPSILON,
    HORIZON,
    HORIZONTAL_SEPARATION,
    VERTICAL_SEPARATION,
    ConflictResult,
    detect_all_conflicts,
    detect_conflict,
)

# Headings: 0 = North, 90 = East, 180 = South, 270 = West.
EAST = 90.0
WEST = 270.0
NORTH = 0.0
SOUTH = 180.0


def flying(
    aircraft_id,
    position=(0.0, 0.0),
    heading=EAST,
    speed=10.0,
    altitude=2000.0,
    state=AircraftState.CRUISE,
    target_altitude=2000.0,
):
    """An airborne aircraft with a single waypoint at its target altitude.

    Giving it a waypoint whose altitude equals the aircraft altitude makes
    ``vertical_rate`` zero, so the vertical test case can be isolated.
    """
    waypoint = Waypoint(
        f"{aircraft_id}_WP", position[0], position[1], target_altitude
    )
    return Aircraft(
        id=aircraft_id,
        state=state,
        position=position,
        heading=heading,
        speed=speed,
        altitude=altitude,
        air_route=[waypoint],
    )


# ----------------------------------------------------------------------
# Independent brute-force oracle (test code only)
# ----------------------------------------------------------------------
def brute_force_conflict(a, b, horizon, h_sep, v_sep, steps=200_000):
    """Return ``True`` if a violation occurs anywhere in the horizon.

    Samples the horizon densely instead of solving a quadratic, so it shares no
    code path with the detector under test.
    """
    vax, vay = velocity_from_heading(a.speed, a.heading)
    vbx, vby = velocity_from_heading(b.speed, b.heading)
    dz0 = a.altitude - b.altitude
    rvz = vertical_rate(a) - vertical_rate(b)
    for i in range(steps + 1):
        t = horizon * i / steps
        dx = (a.position[0] + vax * t) - (b.position[0] + vbx * t)
        dy = (a.position[1] + vay * t) - (b.position[1] + vby * t)
        if math.hypot(dx, dy) <= h_sep and abs(dz0 + rvz * t) <= v_sep:
            return True
    return False


# ----------------------------------------------------------------------
# Constants and construction
# ----------------------------------------------------------------------
def test_default_separation_constants():
    assert HORIZON == 10.0
    assert HORIZONTAL_SEPARATION == 5.0
    assert VERTICAL_SEPARATION == 100.0
    assert EPSILON > 0.0


def test_no_conflict_result_reports_false_and_no_time():
    result = detect_conflict(flying("AC001"), flying("AC002", (100.0, 0.0)))
    assert isinstance(result, ConflictResult)
    assert result.conflict is False
    assert result.time_to_conflict is None
    assert result.horizontal_distance is None
    assert result.predicted_position_a is None
    assert result.predicted_position_b is None


def test_result_keeps_both_aircraft():
    a = flying("AC001")
    b = flying("AC002", (100.0, 0.0))
    result = detect_conflict(a, b)
    assert result.aircraft_a is a
    assert result.aircraft_b is b


def test_detector_does_not_modify_either_aircraft():
    a = flying("AC001")
    b = flying("AC002", (9.0, 0.0), heading=WEST)
    before_a = (a.position, a.altitude, a.speed, a.heading, a.state)
    before_b = (b.position, b.altitude, b.speed, b.heading, b.state)
    detect_conflict(a, b)
    assert (a.position, a.altitude, a.speed, a.heading, a.state) == before_a
    assert (b.position, b.altitude, b.speed, b.heading, b.state) == before_b


# ----------------------------------------------------------------------
# Hand-calculated head-on case
# ----------------------------------------------------------------------
def test_head_on_conflict_is_detected():
    # A at (0,0) flying East at 10; B at (100,0) flying West at 10.
    # They meet at t = 5 in the middle (50, 0).
    a = flying("AC001", (0.0, 0.0), EAST, 10.0)
    b = flying("AC002", (100.0, 0.0), WEST, 10.0)
    result = detect_conflict(a, b)
    assert result.conflict is True
    # Separation <= 5 from t = 4.75 to t = 5.25, so the first violation is 4.75.
    assert result.time_to_conflict == pytest.approx(4.75)
    assert result.horizontal_cpa_time == pytest.approx(5.0)
    assert result.horizontal_distance == pytest.approx(5.0)
    assert result.vertical_distance == pytest.approx(0.0)


def test_head_on_predicted_positions_are_hand_checked():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0)
    b = flying("AC002", (100.0, 0.0), WEST, 10.0)
    result = detect_conflict(a, b)
    t = result.time_to_conflict
    assert result.predicted_position_a == pytest.approx((10.0 * t, 0.0))
    assert result.predicted_position_b == pytest.approx((100.0 - 10.0 * t, 0.0))
    # At 4.75 the two aircraft are 5 units apart.
    assert abs(
        result.predicted_position_a[0] - result.predicted_position_b[0]
    ) == pytest.approx(5.0)


def test_head_on_meeting_point_is_the_cpa():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0)
    b = flying("AC002", (100.0, 0.0), WEST, 10.0)
    result = detect_conflict(a, b)
    t = result.horizontal_cpa_time
    # At the CPA the two aircraft occupy the same point: (50, 0) at t = 5.
    assert a.position[0] + 10.0 * t == pytest.approx(50.0)
    assert b.position[0] - 10.0 * t == pytest.approx(50.0)


# ----------------------------------------------------------------------
# Direction handling
# ----------------------------------------------------------------------
def test_diverging_aircraft_never_conflict():
    # 10 units apart, A flying West and B flying East: the gap only widens.
    a = flying("AC001", (50.0, 0.0), WEST, 10.0)
    b = flying("AC002", (60.0, 0.0), EAST, 10.0)
    result = detect_conflict(a, b)
    assert result.conflict is False


def test_aircraft_already_overlapping_conflict_at_time_zero():
    a = flying("AC001", (50.0, 0.0), EAST, 10.0)
    b = flying("AC002", (51.0, 0.0), EAST, 10.0)
    result = detect_conflict(a, b)
    assert result.conflict is True
    assert result.time_to_conflict == pytest.approx(0.0)
    assert result.horizontal_distance == pytest.approx(1.0)


def test_converging_perpendicular_pair_conflict():
    # A flies East along y = 0, B flies South along x = 10 from (10, 10).
    # They cross exactly at (10, 0) at t = 1.
    a = flying("AC001", (0.0, 0.0), EAST, 10.0)
    b = flying("AC002", (10.0, 10.0), SOUTH, 10.0)
    result = detect_conflict(a, b)
    assert result.conflict is True
    assert result.horizontal_cpa_time == pytest.approx(1.0)
    # The reported distances are measured at time_to_conflict, the first
    # violating instant (where the gap is exactly the 5.0 threshold), not at
    # the CPA where the two aircraft actually touch.
    assert result.horizontal_distance == pytest.approx(
        HORIZONTAL_SEPARATION
    )


def test_parallel_same_direction_far_apart_never_conflict():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0)
    b = flying("AC002", (0.0, 30.0), EAST, 10.0)
    assert detect_conflict(a, b).conflict is False


def test_parallel_same_direction_within_separation_conflict():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0)
    b = flying("AC002", (0.0, 3.0), EAST, 10.0)
    result = detect_conflict(a, b)
    assert result.conflict is True
    assert result.time_to_conflict == pytest.approx(0.0)


def test_slow_catch_up_from_behind_conflicts_late():
    # A 50 units behind, 1 unit/s faster -> closes 1 unit per tick.
    a = flying("AC001", (0.0, 0.0), EAST, 10.0)
    b = flying("AC002", (50.0, 0.0), EAST, 9.0)
    result = detect_conflict(a, b, horizon=100.0)
    assert result.conflict is True
    # Separation 5 at t = 45.
    assert result.time_to_conflict == pytest.approx(45.0)


# ----------------------------------------------------------------------
# Horizon
# ----------------------------------------------------------------------
def test_conflict_beyond_the_horizon_is_not_reported():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0)
    b = flying("AC002", (200.0, 0.0), WEST, 10.0)  # meet at t = 10
    result = detect_conflict(a, b, horizon=5.0)
    assert result.conflict is False
    assert result.horizontal_cpa_time == pytest.approx(5.0)


def test_conflict_exactly_at_the_horizon_is_reported():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0)
    b = flying("AC002", (200.0, 0.0), WEST, 10.0)  # meet exactly at t = 10
    result = detect_conflict(a, b, horizon=10.0)
    assert result.conflict is True
    # They are 5 apart from t = 9.75, which is the first violation.
    assert result.time_to_conflict == pytest.approx(9.75)
    assert result.horizontal_cpa_time == pytest.approx(10.0)


def test_widening_the_horizon_can_only_add_conflicts():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0)
    b = flying("AC002", (200.0, 0.0), WEST, 10.0)
    assert detect_conflict(a, b, horizon=5.0).conflict is False
    assert detect_conflict(a, b, horizon=10.0).conflict is True


# ----------------------------------------------------------------------
# Vertical separation
# ----------------------------------------------------------------------
def test_horizontal_conflict_but_safe_vertically_is_no_conflict():
    # Same crossing point, but 400 altitude units apart, climbing at the
    # shared ALTITUDE_RATE, so the vertical gap never closes.
    a = flying("AC001", (0.0, 0.0), EAST, 10.0, altitude=0.0,
               target_altitude=1000.0)
    b = flying("AC002", (100.0, 0.0), WEST, 10.0, altitude=400.0,
               target_altitude=1400.0)
    result = detect_conflict(a, b)
    assert result.conflict is False


def test_vertical_gap_outside_separation_prevents_conflict():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0, altitude=0.0,
               target_altitude=0.0)
    b = flying("AC002", (1.0, 0.0), WEST, 10.0, altitude=1000.0,
               target_altitude=1000.0)
    # Horizontally 1 unit apart, vertically 1000 apart: safe.
    assert detect_conflict(a, b).conflict is False


def test_vertical_separation_boundary_is_inclusive():
    # Exactly VERTICAL_SEPARATION apart and horizontally coincident: the
    # threshold is met, so a conflict is reported.
    a = flying("AC001", (0.0, 0.0), EAST, 10.0, altitude=0.0,
               target_altitude=0.0)
    b = flying("AC002", (0.0, 0.0), EAST, 0.0, altitude=100.0,
               target_altitude=100.0)
    result = detect_conflict(a, b)
    assert result.conflict is True
    assert result.vertical_distance == pytest.approx(100.0)


def test_vertical_gap_closes_while_horizontal_gap_closes():
    # A climbs at ALTITUDE_RATE from 0 toward 1000; B holds 550.
    # Vertical: 100t - 550 >= -100 -> t >= 4.5.
    # Horizontal: t in [4.75, 5.25].  The overlap starts at 4.75.
    a = flying("AC001", (0.0, 0.0), EAST, 10.0, altitude=0.0,
               target_altitude=1000.0)
    b = flying("AC002", (100.0, 0.0), WEST, 10.0, altitude=550.0,
               target_altitude=550.0)
    result = detect_conflict(a, b)
    assert result.conflict is True
    assert result.time_to_conflict == pytest.approx(4.75)
    # Reported at time_to_conflict: the vertical gap is 550 - 100*4.75 = 75.
    assert result.vertical_distance == pytest.approx(75.0)


def test_vertical_and_horizontal_must_coincide_in_time():
    # Horizontal: they meet at t = 1.5, so the window is [1.25, 1.75].
    # Vertical:   100t - 550 >= -100 -> t in [4.5, 10].
    # The two windows do not overlap, so this is NOT a conflict.
    a = flying("AC001", (0.0, 0.0), EAST, 10.0, altitude=0.0,
               target_altitude=1000.0)
    b = flying("AC002", (30.0, 0.0), WEST, 10.0, altitude=550.0,
               target_altitude=550.0)
    result = detect_conflict(a, b)
    assert result.conflict is False
    assert result.horizontal_cpa_time == pytest.approx(1.5)


def test_vertical_rate_uses_the_shared_airspace_helper():
    a = flying("AC001", altitude=0.0, target_altitude=1000.0)
    b = flying("AC002", altitude=500.0, target_altitude=500.0)
    assert vertical_rate(a) == ALTITUDE_RATE
    assert vertical_rate(b) == 0.0


# ----------------------------------------------------------------------
# Degenerate kinematics
# ----------------------------------------------------------------------
def test_both_aircraft_stationary_but_overlapping_conflict():
    a = flying("AC001", (10.0, 10.0), EAST, 0.0)
    b = flying("AC002", (11.0, 10.0), EAST, 0.0)
    result = detect_conflict(a, b)
    assert result.conflict is True
    assert result.time_to_conflict == pytest.approx(0.0)
    assert result.horizontal_cpa_time is None  # no relative motion


def test_both_aircraft_stationary_and_far_apart_never_conflict():
    a = flying("AC001", (0.0, 0.0), EAST, 0.0)
    b = flying("AC002", (100.0, 0.0), EAST, 0.0)
    result = detect_conflict(a, b)
    assert result.conflict is False
    assert result.horizontal_cpa_time is None


def test_stationary_aircraft_conflicts_with_a_mover():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0)
    b = flying("AC002", (20.0, 0.0), EAST, 0.0)
    result = detect_conflict(a, b)
    assert result.conflict is True
    assert result.time_to_conflict == pytest.approx(1.5)


def test_tiny_relative_velocity_is_handled_without_dividing_by_zero():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0)
    b = flying("AC002", (40.0, 0.0), EAST, 10.0 + 1e-6)
    result = detect_conflict(a, b, horizon=1000.0)
    # Closing at 1e-6 per tick from 40 units: needs ~3.5e7 ticks, far beyond.
    assert result.conflict is False


def test_tiny_relative_velocity_can_still_conflict_eventually():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0)
    b = flying("AC002", (40.0, 0.0), EAST, 9.9999)
    result = detect_conflict(a, b, horizon=1_000_000.0)
    assert result.conflict is True


# ----------------------------------------------------------------------
# Thresholds and parameters
# ----------------------------------------------------------------------
def test_horizontal_separation_boundary_is_inclusive():
    a = flying("AC001", (0.0, 0.0), EAST, 0.0)
    b = flying("AC002", (5.0, 0.0), EAST, 0.0)
    assert detect_conflict(a, b).conflict is True  # exactly 5.0 apart


def test_just_outside_horizontal_separation_is_no_conflict():
    a = flying("AC001", (0.0, 0.0), EAST, 0.0)
    b = flying("AC002", (5.001, 0.0), EAST, 0.0)
    assert detect_conflict(a, b).conflict is False


def test_custom_separation_thresholds_are_honoured():
    a = flying("AC001", (0.0, 0.0), EAST, 0.0)
    b = flying("AC002", (20.0, 0.0), EAST, 0.0)
    # 20 apart: safe at the default 5, conflicting at 25.
    assert detect_conflict(a, b).conflict is False
    assert detect_conflict(a, b, horizontal_separation=25.0).conflict is True


def test_zero_separation_threshold_flags_any_coincidence():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0)
    b = flying("AC002", (100.0, 0.0), WEST, 10.0)
    # With a zero threshold the aircraft must actually touch.
    assert detect_conflict(a, b, horizontal_separation=0.0).conflict is True


def test_negative_horizon_is_rejected():
    a = flying("AC001")
    b = flying("AC002", (100.0, 0.0))
    with pytest.raises(ValueError, match="horizon"):
        detect_conflict(a, b, horizon=-1.0)


def test_negative_separation_is_rejected():
    a = flying("AC001")
    b = flying("AC002", (100.0, 0.0))
    with pytest.raises(ValueError, match="separation"):
        detect_conflict(a, b, horizontal_separation=-1.0)


def test_missing_aircraft_is_rejected():
    a = flying("AC001")
    with pytest.raises(ValueError, match="required"):
        detect_conflict(a, None)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Symmetry
# ----------------------------------------------------------------------
def test_detection_is_symmetric():
    a = flying("AC001", (0.0, 0.0), EAST, 10.0)
    b = flying("AC002", (100.0, 0.0), WEST, 10.0)
    forward = detect_conflict(a, b)
    backward = detect_conflict(b, a)
    assert forward.conflict == backward.conflict
    assert forward.time_to_conflict == pytest.approx(backward.time_to_conflict)
    assert forward.horizontal_distance == pytest.approx(
        backward.horizontal_distance
    )
    assert forward.predicted_position_a == pytest.approx(
        backward.predicted_position_b
    )


# ----------------------------------------------------------------------
# All-pairs scanning
# ----------------------------------------------------------------------
def test_detect_all_conflicts_checks_every_unordered_pair():
    fleet = [
        flying("AC001", (0.0, 0.0), EAST, 10.0),
        flying("AC002", (100.0, 0.0), WEST, 10.0),
        flying("AC003", (0.0, 500.0), EAST, 10.0),
    ]
    results = detect_all_conflicts(fleet)
    assert len(results) == 3  # 3 * 2 / 2
    # (AC001, AC002) conflicts; the two AC003 pairs do not.
    assert results[0].conflict is True
    assert results[1].conflict is False
    assert results[2].conflict is False


def test_detect_all_conflicts_ignores_ground_aircraft():
    fleet = [
        flying("AC001", (0.0, 0.0), EAST, 10.0),
        flying("AC002", (100.0, 0.0), WEST, 10.0),
        Aircraft(id="AC003", state=AircraftState.AT_GATE, position=(0.0, 0.0)),
        Aircraft(
            id="AC004", state=AircraftState.TAXIING_TO_RUNWAY, position=(1.0, 0.0)
        ),
    ]
    results = detect_all_conflicts(fleet)
    assert len(results) == 1
    assert results[0].aircraft_a.id == "AC001"
    assert results[0].aircraft_b.id == "AC002"


def test_detect_all_conflicts_uses_the_airborne_state_set():
    fleet = [flying(f"AC{index:03d}", (0.0, 0.0)) for index in range(1, 4)]
    for aircraft in fleet:
        assert aircraft.state in AIRBORNE_STATES
    assert len(detect_all_conflicts(fleet)) == 3


def test_detect_all_conflicts_on_a_single_aircraft_is_empty():
    assert detect_all_conflicts([flying("AC001")]) == []


def test_detect_all_conflicts_on_an_empty_fleet_is_empty():
    assert detect_all_conflicts([]) == []


# ----------------------------------------------------------------------
# Cross-check against the independent brute-force oracle
# ----------------------------------------------------------------------
SCENARIOS = [
    # (A position, A heading, A speed, A altitude, target altitude,
    #  B position, B heading, B speed, B altitude, target altitude, horizon)
    ((0.0, 0.0), EAST, 10.0, 2000.0, 2000.0,
     (100.0, 0.0), WEST, 10.0, 2000.0, 2000.0, 10.0),
    ((0.0, 0.0), EAST, 10.0, 2000.0, 2000.0,
     (200.0, 0.0), WEST, 10.0, 2000.0, 2000.0, 5.0),
    ((0.0, 0.0), EAST, 12.0, 0.0, 3000.0,
     (60.0, 0.0), WEST, 8.0, 200.0, 200.0, 15.0),
    ((0.0, 0.0), NORTH, 5.0, 500.0, 500.0,
     (0.0, 30.0), SOUTH, 5.0, 500.0, 500.0, 10.0),
    ((-20.0, -20.0), 30.0, 7.0, 1000.0, 4000.0,
     (40.0, 35.0), 210.0, 7.0, 2500.0, 2500.0, 20.0),
    ((0.0, 0.0), EAST, 10.0, 0.0, 1000.0,
     (3.0, 0.0), WEST, 10.0, 150.0, 150.0, 10.0),
    ((0.0, 0.0), EAST, 0.0, 100.0, 100.0,
     (2.0, 0.0), EAST, 0.0, 100.0, 100.0, 10.0),
    ((0.0, 0.0), EAST, 10.0, 2000.0, 2000.0,
     (0.0, 5.0), EAST, 10.0, 2000.0, 2000.0, 10.0),
]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_analytical_detector_agrees_with_the_brute_force_oracle(scenario):
    (
        a_pos, a_heading, a_speed, a_alt, a_target,
        b_pos, b_heading, b_speed, b_alt, b_target, horizon,
    ) = scenario
    a = flying("AC001", a_pos, a_heading, a_speed, a_alt,
               target_altitude=a_target)
    b = flying("AC002", b_pos, b_heading, b_speed, b_alt,
               target_altitude=b_target)

    analytical = detect_conflict(a, b, horizon=horizon)
    brute = brute_force_conflict(
        a, b, horizon, HORIZONTAL_SEPARATION, VERTICAL_SEPARATION
    )
    assert analytical.conflict == brute

    if analytical.conflict:
        # The reported time must be the FIRST violating instant, so sampling
        # just before it must find no violation.
        t = analytical.time_to_conflict
        assert t >= 0.0 and t <= horizon
        vax, vay = velocity_from_heading(a.speed, a.heading)
        vbx, vby = velocity_from_heading(b.speed, b.heading)
        dz0 = a.altitude - b.altitude
        rvz = vertical_rate(a) - vertical_rate(b)
        just_before = t - 1e-6
        if just_before > 0.0:
            dx = (a.position[0] + vax * just_before) - (
                b.position[0] + vbx * just_before
            )
            dy = (a.position[1] + vay * just_before) - (
                b.position[1] + vby * just_before
            )
            assert not (
                math.hypot(dx, dy) <= HORIZONTAL_SEPARATION
                and abs(dz0 + rvz * just_before) <= VERTICAL_SEPARATION
            )
