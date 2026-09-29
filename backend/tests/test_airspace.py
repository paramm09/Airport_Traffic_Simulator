"""Phase 2D tests: continuous airspace movement along waypoint routes.

The airspace is a flat plane with a deterministic movement model:

    position += speed * dt toward the current waypoint
    altitude  converges toward the current waypoint altitude at ALTITUDE_RATE

Heading follows the shared convention ``degrees(atan2(dx, dy)) % 360`` with
0 = North, 90 = East, 180 = South, 270 = West, which Phase 2E reuses for its
kinematic conflict prediction.
"""

import math
import pytest

from backend.data.airspace_waypoints import (
    ARR_EAST,
    CRUISE_NORTH,
    DEP_EAST,
)
from backend.models.aircraft import Aircraft, AircraftState
from backend.models.waypoint import Waypoint
from backend.simulation.airspace import (
    AIRBORNE_STATES,
    ALTITUDE_RATE,
    AirspaceStepOutcome,
    airspace_step,
    heading_from_direction,
    velocity_from_heading,
    vertical_rate,
)


def airborne(route, state=AircraftState.TAKEOFF, speed=10.0, **kwargs):
    """An aircraft ready to fly through ``route``."""
    return Aircraft(
        id="AC001",
        state=state,
        air_route=list(route),
        speed=speed,
        **kwargs,
    )


def test_altitude_rate_constant():
    assert ALTITUDE_RATE == 100.0


def test_airborne_states_cover_the_airborne_chain():
    expected = {
        AircraftState.TAKEOFF,
        AircraftState.CLIMB,
        AircraftState.CRUISE,
        AircraftState.DESCENT,
        AircraftState.APPROACH,
    }
    assert AIRBORNE_STATES == expected


def test_landing_is_not_airborne():
    assert AircraftState.LANDING not in AIRBORNE_STATES


def test_heading_north_is_zero():
    assert heading_from_direction(0.0, 1.0) == 0.0


def test_heading_east_is_ninety():
    assert heading_from_direction(1.0, 0.0) == 90.0


def test_heading_south_is_one_eighty():
    assert heading_from_direction(0.0, -1.0) == 180.0


def test_heading_west_is_two_seventy():
    assert heading_from_direction(-1.0, 0.0) == 270.0


def test_heading_is_wrapped_into_degrees():
    assert heading_from_direction(1.0, 1.0) == 45.0
    # atan2 gives -90 for straight west, which must wrap to 270.
    assert heading_from_direction(-1.0, 0.0) == 270.0


def test_velocity_matches_heading_maths():
    vx, vy = velocity_from_heading(10.0, 90.0)  # due east
    assert math.isclose(vx, 10.0, abs_tol=1e-9)
    assert math.isclose(vy, 0.0, abs_tol=1e-9)

    vx, vy = velocity_from_heading(5.0, 0.0)  # due north
    assert math.isclose(vx, 0.0, abs_tol=1e-9)
    assert math.isclose(vy, 5.0, abs_tol=1e-9)


def test_velocity_round_trip_with_heading_from_direction():
    dx, dy = 3.0, 4.0
    heading = heading_from_direction(dx, dy)
    vx, vy = velocity_from_heading(10.0, heading)
    # The velocity must point along (3, 4).
    dot = vx * dx + vy * dy
    assert dot > 0.0


def test_aircraft_position_moves_toward_the_first_waypoint():
    aircraft = airborne([ARR_EAST], speed=5.0)  # (10, 0) from (0,0)
    outcome = airspace_step(aircraft, dt=1.0)
    assert outcome is AirspaceStepOutcome.MOVED
    # speed 5, one tick -> 5 units toward (10,0).
    assert aircraft.position == (5.0, 0.0)
    assert aircraft.air_route_index == 0


def test_reaching_the_waypoint_in_one_tick_completes_the_route():
    aircraft = airborne([ARR_EAST])  # (10, 0), altitude 2000
    outcome = airspace_step(aircraft, dt=1.0)  # speed 10 covers 10 units.
    assert outcome is AirspaceStepOutcome.ROUTE_COMPLETE
    assert aircraft.position == (10.0, 0.0)
    assert aircraft.air_route_index == 1


def test_aircraft_moves_partway_when_distance_is_large():
    aircraft = airborne([Waypoint("FAR", 0.0, 10.0, 1000.0)], speed=5.0)
    airspace_step(aircraft, dt=1.0)
    assert aircraft.air_route_index == 0
    assert aircraft.position == pytest.approx((0.0, 5.0))


def test_reaching_the_final_waypoint_completes_the_route():
    aircraft = airborne([Waypoint("W", 10.0, 0.0, 1000.0)], speed=5.0)
    airspace_step(aircraft, dt=1.0)  # 5 units -> (5, 0)
    outcome = airspace_step(aircraft, dt=1.0)  # 5 units -> (10, 0)
    assert outcome is AirspaceStepOutcome.ROUTE_COMPLETE
    assert aircraft.position == (10.0, 0.0)
    assert aircraft.air_route_index == len(aircraft.air_route)
    assert aircraft.next_waypoint is None


def test_leftover_distance_carries_into_the_next_waypoint():
    aircraft = airborne(
        [Waypoint("A", 2.0, 0.0, 1000.0), Waypoint("B", 6.0, 0.0, 1000.0)],
        speed=5.0,
    )
    airspace_step(aircraft, dt=1.0)  # move 5: 2 to A, leftover 3 toward B
    assert aircraft.air_route_index == 1
    assert aircraft.position == (5.0, 0.0)  # 3 of the 4 units to B


def test_move_route_index_advances_per_waypoint():
    aircraft = airborne(
        [
            Waypoint("A", 2.0, 0.0, 1000.0),
            Waypoint("B", 2.0, 4.0, 1000.0),
            Waypoint("C", 2.0, 8.0, 1000.0),
        ],
        speed=4.0,
    )
    outcomes = [airspace_step(aircraft, dt=1.0) for _ in range(3)]
    assert outcomes == [
        AirspaceStepOutcome.MOVED,
        AirspaceStepOutcome.MOVED,
        AirspaceStepOutcome.ROUTE_COMPLETE,
    ]
    assert aircraft.position == (2.0, 8.0)
    assert aircraft.air_route_index == 3


def test_altitude_converges_on_the_target_at_altitude_rate():
    aircraft = airborne([DEP_EAST])  # altitude 1000
    airspace_step(aircraft, dt=1.0)
    assert aircraft.altitude == ALTITUDE_RATE  # 100 of 1000 in one tick


def test_altitude_does_not_overshoot_the_target():
    aircraft = airborne(
        [Waypoint("CLOSE", 10.0, 0.0, 50.0)],
        altitude=0.0,
        speed=5.0,
    )
    airspace_step(aircraft, dt=1.0)
    assert aircraft.altitude == pytest.approx(50.0)


def test_altitude_climbs_and_descends():
    aircraft = airborne(
        [Waypoint("UP", 0.0, 10.0, 500.0), Waypoint("DOWN", 0.0, 20.0, 200.0)],
        altitude=0.0,
    )
    airspace_step(aircraft, dt=1.0)
    assert aircraft.altitude == pytest.approx(ALTITUDE_RATE)
    airspace_step(aircraft, dt=1.0)
    # climbs until reaching 500 then descends toward 200.
    assert aircraft.altitude <= 500.0


def test_heading_is_set_to_point_at_the_waypoint():
    aircraft = airborne([Waypoint("E", 10.0, 0.0, 1000.0)], speed=5.0)
    airspace_step(aircraft, dt=1.0)
    assert aircraft.heading == pytest.approx(90.0)


def test_vertical_rate_positive_when_climbing():
    aircraft = airborne([Waypoint("UP", 0.0, 10.0, 2000.0)], altitude=0.0)
    assert vertical_rate(aircraft) == ALTITUDE_RATE


def test_vertical_rate_negative_when_descending():
    aircraft = airborne([Waypoint("DN", 0.0, 10.0, 0.0)], altitude=500.0)
    assert vertical_rate(aircraft) == -ALTITUDE_RATE


def test_vertical_rate_zero_at_the_target_altitude():
    aircraft = airborne([Waypoint("FLAT", 0.0, 10.0, 1000.0)], altitude=1000.0)
    assert vertical_rate(aircraft) == 0.0


def test_vertical_rate_zero_without_an_air_route():
    aircraft = Aircraft(id="AC001", state=AircraftState.CRUISE, altitude=100.0)
    assert vertical_rate(aircraft) == 0.0


def test_speed_zero_does_not_move_the_aircraft():
    aircraft = airborne([Waypoint("S", 5.0, 0.0, 1000.0)], speed=0.0)
    airspace_step(aircraft, dt=1.0)
    assert aircraft.position == (0.0, 0.0)
    assert aircraft.air_route_index == 0


def test_dt_must_be_positive():
    aircraft = airborne([DEP_EAST])
    with pytest.raises(ValueError, match="positive"):
        airspace_step(aircraft, dt=0.0)
    with pytest.raises(ValueError, match="positive"):
        airspace_step(aircraft, dt=-1.0)


def test_wrong_state_does_not_move_the_aircraft():
    aircraft = airborne([DEP_EAST], state=AircraftState.AT_GATE)
    with pytest.raises(ValueError, match="airborne"):
        airspace_step(aircraft, dt=1.0)
    assert aircraft.position == (0.0, 0.0)
    assert aircraft.air_route_index == 0


def test_empty_air_route_is_rejected():
    aircraft = Aircraft(
        id="AC001", state=AircraftState.TAKEOFF, air_route=[]
    )
    with pytest.raises(ValueError, match="no air route"):
        airspace_step(aircraft, dt=1.0)


def test_big_dt_crosses_several_waypoints():
    aircraft = airborne(
        [
            Waypoint("A", 1.0, 0.0, 1000.0),
            Waypoint("B", 2.0, 0.0, 1000.0),
            Waypoint("C", 3.0, 0.0, 1000.0),
        ],
        speed=10.0,
    )
    airspace_step(aircraft, dt=1.0)  # 10 units covers the whole route
    assert aircraft.position == (3.0, 0.0)
    assert aircraft.air_route_index == 3
    assert aircraft.next_waypoint is None


def test_route_complete_transitions_to_next_lifecycle_state():
    # The aircraft starts on DEP_EAST (same position), so the step finishes
    # immediately and advances the lifecycle chain.
    aircraft = airborne([DEP_EAST], state=AircraftState.TAKEOFF)
    outcome = airspace_step(aircraft, dt=1.0)
    assert outcome is AirspaceStepOutcome.ROUTE_COMPLETE
    assert aircraft.state is AircraftState.CLIMB

    aircraft = airborne([DEP_EAST], state=AircraftState.CRUISE)
    aircraft.position = (0.0, 0.0)  # already on DEP_EAST -> immediate complete
    outcome = airspace_step(aircraft, dt=1.0)
    assert outcome is AirspaceStepOutcome.ROUTE_COMPLETE
    assert aircraft.state is AircraftState.DESCENT


def test_waypoint_rejection_of_an_empty_id():
    with pytest.raises(ValueError, match="id"):
        Waypoint("", 0.0, 0.0, 1000.0)


def test_negative_altitude_waypoint_is_rejected():
    with pytest.raises(ValueError, match="altitude"):
        Waypoint("BAD", 0.0, 0.0, -100.0)


def test_aircraft_rejects_a_non_waypoint_route_entry():
    with pytest.raises(ValueError, match="Waypoint"):
        Aircraft(
            id="AC001",
            state=AircraftState.CRUISE,
            air_route=["DEP_EAST"],  # type: ignore[list-item]
        )


def test_aircraft_route_index_must_not_be_negative():
    with pytest.raises(ValueError, match="air_route_index"):
        Aircraft(
            id="AC001",
            state=AircraftState.CRUISE,
            air_route=[DEP_EAST],
            air_route_index=-1,
        )


def test_aircraft_moves_along_the_standard_airspace():
    aircraft = airborne([DEP_EAST, CRUISE_NORTH, ARR_EAST], speed=5.0)
    airspace_step(aircraft, dt=1.0)
    # DEP_EAST is at (0,0) (distance 0), so it is skipped; the aircraft now
    # flies 5 of the 10 units toward CRUISE_NORTH (0,10).
    assert aircraft.position == pytest.approx((0.0, 5.0))
    assert aircraft.air_route_index == 1
    assert aircraft.next_waypoint is not None
    airspace_step(aircraft, dt=1.0)
    # 5 more units snaps onto CRUISE_NORTH.
    assert aircraft.position == pytest.approx((0.0, 10.0))
    assert aircraft.air_route_index == 2
    assert aircraft.next_waypoint is not None


# ----------------------------------------------------------------------
# Heading stays consistent with position across waypoint advancement
# ----------------------------------------------------------------------
def test_a_single_waypoint_leg_sets_the_heading():
    """The ordinary case: one waypoint, reached part-way."""
    aircraft = airborne([Waypoint("A", 0.0, 30.0, 1000.0)], speed=10.0)
    airspace_step(aircraft, dt=1.0)
    assert aircraft.position == pytest.approx((0.0, 10.0))
    assert aircraft.heading == pytest.approx(0.0)  # due North


def test_crossing_a_waypoint_leaves_the_heading_on_the_new_leg():
    """Crossing one waypoint must not leave the previous leg's heading.

    The route turns 90 degrees at A, so a stale 270 (West) would point the
    aircraft back the way it came instead of at B.
    """
    aircraft = airborne(
        [
            Waypoint("A", 10.0, 0.0, 1000.0),
            Waypoint("B", 10.0, 40.0, 1000.0),
        ],
        speed=10.0,
    )
    airspace_step(aircraft, dt=1.0)  # reaches A exactly
    assert aircraft.position == pytest.approx((10.0, 0.0))
    assert aircraft.air_route_index == 1
    assert aircraft.heading == pytest.approx(0.0)  # North, toward B


def test_a_large_dt_crossing_several_waypoints_keeps_the_last_leg_heading():
    """One call that swallows the whole route keeps the *final* leg's heading.

    All three legs are collinear here, so this checks the direction is right
    rather than merely non-stale.
    """
    aircraft = airborne(
        [
            Waypoint("A", 10.0, 0.0, 1000.0),
            Waypoint("B", 20.0, 0.0, 1000.0),
            Waypoint("C", 30.0, 0.0, 1000.0),
        ],
        speed=10.0,
    )
    outcome = airspace_step(aircraft, dt=10.0)  # 100 units, whole route
    assert outcome is AirspaceStepOutcome.ROUTE_COMPLETE
    assert aircraft.position == pytest.approx((30.0, 0.0))
    assert aircraft.heading == pytest.approx(90.0)  # East, the final leg


def test_crossing_waypoints_turns_the_heading_onto_the_remaining_leg():
    """The case that matters most: a big dt that stops mid-route after turning.

    10 units per tick, so dt=2.5 buys 25: 10 to reach A, 10 more to reach B, and
    the last 5 part-way up the B->C leg. The aircraft is still airborne with C
    ahead, so its heading must describe B->C and not the A->B leg it just left.
    """
    aircraft = airborne(
        [
            Waypoint("A", 10.0, 0.0, 1000.0),
            Waypoint("B", 20.0, 0.0, 1000.0),
            Waypoint("C", 20.0, 100.0, 1000.0),
        ],
        speed=10.0,
    )
    outcome = airspace_step(aircraft, dt=2.5)
    assert outcome is AirspaceStepOutcome.MOVED
    assert aircraft.air_route_index == 2
    assert aircraft.position == pytest.approx((20.0, 5.0))
    # The remaining leg is due North, not the East it was flying a moment ago.
    assert aircraft.heading == pytest.approx(0.0)


def test_the_heading_always_points_at_the_next_waypoint():
    """The invariant, checked over a multi-leg route and a variety of dt."""
    route = [
        Waypoint("A", 10.0, 0.0, 1000.0),
        Waypoint("B", 30.0, 20.0, 1000.0),
        Waypoint("C", -5.0, 60.0, 1000.0),
        Waypoint("D", 40.0, -30.0, 1000.0),
    ]
    for dt in (0.3, 1.0, 2.5, 7.0, 13.0):
        aircraft = airborne(route, speed=10.0)
        for _ in range(6):
            outcome = airspace_step(aircraft, dt=dt)
            if outcome is AirspaceStepOutcome.ROUTE_COMPLETE:
                break
            assert aircraft.next_waypoint is not None
            expected = heading_from_direction(
                aircraft.next_waypoint[0] - aircraft.position[0],
                aircraft.next_waypoint[1] - aircraft.position[1],
            )
            assert aircraft.heading == pytest.approx(expected), (dt, aircraft)


def test_a_completed_air_route_keeps_the_direction_of_the_final_leg():
    """Documented behaviour: the final heading is the last leg's direction."""
    aircraft = airborne(
        [
            Waypoint("A", 0.0, 30.0, 1000.0),  # flown due North...
            Waypoint("B", 30.0, 0.0, 1000.0),  # ...then South-East
        ],
        speed=10.0,
    )
    # 100 units covers both the 30 to A and the 42.4 from A to B.
    outcome = airspace_step(aircraft, dt=10.0)
    assert outcome is AirspaceStepOutcome.ROUTE_COMPLETE
    assert aircraft.next_waypoint is None
    # A->B runs exactly South-East, and that is the direction actually flown, so
    # it is the heading that is preserved.
    assert aircraft.heading == pytest.approx(135.0)


def test_a_completed_air_route_at_the_last_waypoint_keeps_a_real_heading():
    """Even a route consumed entirely by distance-0 waypoints keeps a heading."""
    aircraft = airborne(
        [Waypoint("A", 0.0, 0.0, 1000.0), Waypoint("B", 0.0, 0.0, 1000.0)],
        speed=10.0,
    )
    outcome = airspace_step(aircraft, dt=1.0)
    assert outcome is AirspaceStepOutcome.ROUTE_COMPLETE
    assert aircraft.next_waypoint is None
    # Degenerate route, so there is no direction to speak of; what matters is
    # that it is a real number Phase 2E can use.
    assert math.isfinite(aircraft.heading)


def test_the_velocity_cpa_will_use_matches_the_current_heading():
    """The point of the fix: CPA's velocity must match where the aircraft goes.

    ``detect_conflict`` derives the aircraft's velocity from its heading. If the
    heading described a leg the aircraft had already left, CPA would predict
    motion the simulation would never perform.
    """
    from backend.simulation.conflict import detect_conflict

    aircraft = airborne(
        [
            Waypoint("A", 10.0, 0.0, 2000.0),
            Waypoint("B", 20.0, 0.0, 2000.0),
            Waypoint("C", 20.0, 100.0, 2000.0),
        ],
        speed=10.0,
        altitude=2000.0,
        state=AircraftState.CRUISE,
    )
    airspace_step(aircraft, dt=2.5)  # 25 units: crosses A and B, stops on B->C
    assert aircraft.position == pytest.approx((20.0, 5.0))
    assert aircraft.heading == pytest.approx(0.0)  # due North toward C

    vx, vy = velocity_from_heading(aircraft.speed, aircraft.heading)
    # A unit step must move the aircraft the way the reported velocity says.
    next_position = (aircraft.position[0] + vx, aircraft.position[1] + vy)
    airspace_step(aircraft, dt=1.0)
    assert aircraft.position == pytest.approx(next_position, abs=1e-9)

    # And CPA is handed that same, self-consistent aircraft. AC001 is now at
    # (20, 15) heading North and AC002 is 45 units ahead heading South, so they
    # close at 20 units per tick and reach the 5-unit separation at t = 2.0.
    other = Aircraft(
        id="AC002",
        state=AircraftState.CRUISE,
        position=(20.0, 60.0),
        heading=180.0,  # due South, straight at AC001
        speed=10.0,
        altitude=2000.0,
    )
    assert aircraft.position == pytest.approx((20.0, 15.0))
    conflict = detect_conflict(aircraft, other)
    assert conflict.conflict is True
    assert conflict.time_to_conflict == pytest.approx(2.0)
