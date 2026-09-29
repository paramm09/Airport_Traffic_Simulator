"""Continuous airspace movement of an aircraft along its waypoint route.

Phase 2B moves aircraft *on the ground* edge by edge along a Dijkstra route;
Phase 2D moves aircraft *in the air* waypoint to waypoint in continuous
(x, y, altitude) space. The module deliberately contains no shortest-path or
conflict logic — it only answers *"how far has the airborne aircraft travelled
toward its next waypoint?"*.

Movement model (deliberately simple, no real physics):

- ``position`` moves toward ``air_route[air_route_index]`` by ``speed * dt``
  distance units, snapping to the waypoint on arrival and carrying any leftover
  distance into the following waypoint (a mile of freedom, but deterministic).
- ``altitude`` moves toward the *current target waypoint's* altitude at
  ``ALTITUDE_RATE * dt``, clamped so it never overshoots the target altitude.
- ``heading`` is derived from the direction to the aircraft's **next** waypoint
  so that Phase 2E's kinematic model (``vx = speed * sin(heading)``, ``vy =
  speed * cos(heading)``) stays consistent with the movement. It is re-derived
  at the end of every call, so a large ``dt`` that crosses several waypoints
  still leaves position, heading and therefore velocity describing the same
  leg. When the air route is completed the heading is left as the direction of
  the final leg, which is the direction the aircraft actually flew.

Heading convention (0 = North, 90 = East, 180 = South, 270 = West):

    heading = degrees(atan2(dx, dy)) % 360
    vx = speed * sin(radians(heading))
    vy = speed * cos(radians(heading))
"""

from __future__ import annotations

import enum
import math

from backend.models.aircraft import Aircraft, AircraftState

# Rate at which an aircraft's altitude converges on its target waypoint altitude
# (altitude units per simulation tick).
ALTITUDE_RATE = 100.0

# States in which airspace movement is legal (the airborne part of the
# lifecycle chain, up to but excluding the terminal LANDING state which is
# handled by the ground phase).
AIRBORNE_STATES: frozenset[AircraftState] = frozenset(
    {
        AircraftState.TAKEOFF,
        AircraftState.CLIMB,
        AircraftState.CRUISE,
        AircraftState.DESCENT,
        AircraftState.APPROACH,
    }
)

# Lifecycle progression when an air route is completed.
_NEXT_AIRBORNE_STATE: dict[AircraftState, AircraftState] = {
    AircraftState.TAKEOFF: AircraftState.CLIMB,
    AircraftState.CLIMB: AircraftState.CRUISE,
    AircraftState.CRUISE: AircraftState.DESCENT,
    AircraftState.DESCENT: AircraftState.APPROACH,
    AircraftState.APPROACH: AircraftState.LANDING,
}


class AirspaceStepOutcome(enum.Enum):
    """Result of one :func:`airspace_step` call."""

    MOVED = enum.auto()
    """The aircraft advanced through the airspace (mode achieved)."""

    ROUTE_COMPLETE = enum.auto()
    """The final waypoint was reached; the aircraft transitioned to the next
    lifecycle state (e.g. ``TAKEOFF -> CLIMB``)."""


def heading_from_direction(dx: float, dy: float) -> float:
    """Return the compass heading (0=N, 90=E, 180=S, 270=W) of ``(dx, dy)``."""
    return math.degrees(math.atan2(dx, dy)) % 360.0


def velocity_from_heading(speed: float, heading: float) -> tuple[float, float]:
    """Return the ``(vx, vy)`` vector for ``speed`` along a compass ``heading``.

    Uses the same east/north convention as :func:`heading_from_direction` so
    that a heading derived from a direction produces a velocity pointing along
    that exact direction.
    """
    theta = math.radians(heading)
    return (speed * math.sin(theta), speed * math.cos(theta))


def vertical_rate(aircraft: Aircraft) -> float:
    """Return the altitude rate toward the aircraft's current target waypoint.

    ``+ALTITUDE_RATE`` while climbing, ``-ALTITUDE_RATE`` while descending and
    ``0.0`` when the aircraft has no air route or already sits at the target
    altitude. Phase 2E uses this as ``vz`` in its vertical CPA computation.
    """
    if not aircraft.air_route:
        return 0.0
    target = aircraft.air_route[aircraft.air_route_index]
    if target.altitude > aircraft.altitude:
        return ALTITUDE_RATE
    if target.altitude < aircraft.altitude:
        return -ALTITUDE_RATE
    return 0.0


def airspace_step(
    aircraft: Aircraft, dt: float = 1.0
) -> AirspaceStepOutcome:
    """Advance ``aircraft`` by ``speed * dt`` units through the airspace.

    Args:
        aircraft: The moving aircraft. It must be in an airborne state and have
            a non-empty ``air_route``.
        dt: Time step in simulation ticks. Must be positive; movement is
            ``speed * dt`` distance units.

    Returns:
        :attr:`~AirspaceStepOutcome.MOVED` if the aircraft advanced,
        :attr:`~AirspaceStepOutcome.ROUTE_COMPLETE` when the final waypoint was
        reached (the aircraft then transitions to the next lifecycle state).

    Raises:
        ValueError: if ``dt`` is not positive, the aircraft is not in an
            airborne state, or the aircraft has no air route.

    The airborne lifecycle chain is ``TAKEOFF -> CLIMB -> CRUISE -> DESCENT ->
    APPROACH -> LANDING``; completing the route moves the aircraft one step
    along it. ``LANDING`` itself is not considered airborne.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt!r}")
    if aircraft.state not in AIRBORNE_STATES:
        raise ValueError(
            f"aircraft {aircraft.id!r} is in {aircraft.state.name}, expected "
            "an airborne state to advance"
        )
    if not aircraft.air_route:
        raise ValueError(f"aircraft {aircraft.id!r} has no air route")

    remaining = aircraft.speed * dt

    while remaining > 0.0 and aircraft.air_route_index < len(aircraft.air_route):
        target = aircraft.air_route[aircraft.air_route_index]
        dx = target.x - aircraft.position[0]
        dy = target.y - aircraft.position[1]
        distance = math.hypot(dx, dy)

        if distance == 0.0:
            aircraft.air_route_index += 1
            continue

        if distance <= remaining:
            aircraft.position = (target.x, target.y)
            # Record the direction of the leg just flown. This matters when a
            # large dt consumes the rest of the air route in this one call: it
            # leaves the aircraft with the heading of the *final* leg rather
            # than whatever the leg before it happened to be.
            aircraft.heading = heading_from_direction(dx, dy)
            remaining -= distance
            aircraft.air_route_index += 1
        else:
            aircraft.heading = heading_from_direction(dx, dy)
            fraction = remaining / distance
            aircraft.position = (
                aircraft.position[0] + dx * fraction,
                aircraft.position[1] + dy * fraction,
            )
            remaining = 0.0

    # Altitude converges on the current target waypoint's altitude.
    target_index = min(aircraft.air_route_index, len(aircraft.air_route) - 1)
    target_altitude = aircraft.air_route[target_index].altitude
    altitude_gap = target_altitude - aircraft.altitude
    if altitude_gap != 0.0:
        step = ALTITUDE_RATE * dt
        if abs(altitude_gap) <= step:
            aircraft.altitude = target_altitude
        else:
            aircraft.altitude += step if altitude_gap > 0 else -step

    if aircraft.air_route_index >= len(aircraft.air_route):
        # The air route is finished, so there is no next waypoint to point at.
        # The heading is deliberately left as the direction of the final leg
        # (recorded above), because that is the direction the aircraft actually
        # flew and therefore the direction any velocity derived from it should
        # report.
        aircraft.next_waypoint = None
        next_state = _NEXT_AIRBORNE_STATE.get(aircraft.state)
        if next_state is not None and aircraft.can_transition_to(next_state):
            aircraft.transition_to(next_state)
        return AirspaceStepOutcome.ROUTE_COMPLETE

    target = aircraft.air_route[aircraft.air_route_index]
    aircraft.next_waypoint = (target.x, target.y)
    # Crossing one or more waypoints in a single call leaves the aircraft on a
    # later leg, so the heading is re-derived from where it now is to where it
    # is going. Without this, the position would describe the new leg while the
    # heading (and hence the velocity Phase 2E's CPA predicts with) still
    # described the previous one.
    aircraft.heading = heading_from_direction(
        target.x - aircraft.position[0], target.y - aircraft.position[1]
    )
    return AirspaceStepOutcome.MOVED
