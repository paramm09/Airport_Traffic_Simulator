"""Phase 2F: deterministic conflict resolution by HOLD instructions.

Phase 2E answers *"do these two aircraft conflict?"* analytically. This module
answers *"given a set of predicted conflicts, who must yield, and until when?"*.

The resolution strategy is deliberately the simplest one that is still
defensible in an educational simulator: **the lower-priority aircraft is told
to HOLD at its current position for a fixed number of ticks.** Nothing else
changes. In particular this module:

  * never moves an aircraft (no position, heading, speed or altitude writes);
  * never changes an aircraft's lifecycle state;
  * never re-plans a taxi route (that is Phase 2G's job, and it is triggered
    separately for blocked taxi edges, not for air conflicts);
  * never touches the conflict detector, so the detector stays a pure function.

Because a hold only gates movement, a held aircraft is eventually released by
plain time passing: the caller stops gating it once
``current_tick >= aircraft.holding_until``.

Priority
--------
The runway scheduler (Phase 2B) already owns the notion of who outranks whom,
via :data:`PRIORITY_CLASS` over :class:`RunwayOperation`. Conflict resolution
reuses that policy rather than inventing a second, contradictory one, so the
same "landings outrank takeoffs" rule is visible in both places.

An aircraft's operation is *inferred* from its lifecycle state:

    DESCENT / APPROACH / LANDING      -> RunwayOperation.LANDING
    every other lifecycle state       -> RunwayOperation.TAKEOFF

The full ordering, weakest key first, is therefore::

    (priority_class, request_time, aircraft_id)

``PRIORITY_CLASS`` is reused verbatim, so a landing (class 0) outranks a takeoff
(class 1). Within one class the tie-breakers mirror
:meth:`RunwayScheduler.select_next` so the two subsystems agree on ordering:

  1. ``priority_class`` — landing (0) beats takeoff (1);
  2. ``request_time``   — the *stronger* aircraft is the one that requested
     earlier, so the *weaker* one is the later requester and gets held;
  3. ``aircraft_id``    — a final deterministic tie-breaker. Where the two
     ordering keys are otherwise equal there is no aviation reason to prefer
     either aircraft, so the resolver breaks the tie *mechanically and
     symmetrically*: the **lexicographically greater** id is held and the
     smaller id keeps flying. This mirrors the runway queue, where the
     lexicographically smaller id is served first.

Both resolutions are therefore deterministic and require no randomness.

Multiple simultaneous conflicts
-------------------------------
Several conflicts can be predicted in the same tick, and the same aircraft can
appear in more than one of them. Conflicts are handled **most urgent first**:

    sort key = (time_to_conflict, min(aircraft_id), max(aircraft_id))

so the conflict closest to actually happening is resolved before a
further-away one, ties break on the sorted id pair, and the ordering is total
and stable. Because the resolver only ever writes ``holding_until`` — taking
the ``max`` with any existing hold — applying the actions in this order is
order-independent for the final state: a weaker aircraft accumulates the
longest applicable hold instead of overwriting a stronger one with a shorter
one. Cost is O(C log C) for C conflicts, dominated by the sort.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from backend.models.aircraft import Aircraft, AircraftState
from backend.simulation.conflict import ConflictResult
from backend.simulation.runway_scheduler import PRIORITY_CLASS, RunwayOperation

# A HOLD instruction freezes an aircraft for this many simulation ticks.
HOLD_DURATION = 5

# Lifecycle states that belong to the arrival pipeline. Anything else is
# treated as a departure for conflict-resolution priority purposes.
ARRIVAL_STATES: frozenset[AircraftState] = frozenset(
    {
        AircraftState.DESCENT,
        AircraftState.APPROACH,
        AircraftState.LANDING,
    }
)


class ResolutionAction(enum.Enum):
    """What the resolver decided to do about one conflict."""

    HOLD_LOWER_PRIORITY = enum.auto()
    NO_ACTION = enum.auto()


@dataclass(frozen=True)
class ResolutionResult:
    """The outcome of resolving a single :class:`ConflictResult`."""

    conflict: ConflictResult
    action: ResolutionAction
    reason: str
    held_aircraft: Aircraft | None = None
    holding_until: int | None = None


def priority_key(aircraft: Aircraft) -> tuple[int, int, str]:
    """Return the total ordering key for ``aircraft``; smaller is stronger.

    Reuses :data:`PRIORITY_CLASS` from the runway scheduler so both subsystems
    share one priority policy, and adds the two deterministic tie-breakers used
    by the runway heap (``request_time``, then ``aircraft_id``).
    """
    return (PRIORITY_CLASS[operation_for(aircraft)], aircraft.request_time,
            aircraft.id)


def operation_for(aircraft: Aircraft) -> RunwayOperation:
    """Infer the runway operation an aircraft's lifecycle state implies."""
    if aircraft.state in ARRIVAL_STATES:
        return RunwayOperation.LANDING
    return RunwayOperation.TAKEOFF


def is_holding(aircraft: Aircraft, current_tick: int) -> bool:
    """Return ``True`` while ``aircraft`` must not move at ``current_tick``."""
    if aircraft.holding_until is None:
        return False
    return current_tick < aircraft.holding_until


def choose_aircraft_to_hold(
    first: Aircraft, second: Aircraft
) -> tuple[Aircraft, Aircraft]:
    """Return ``(to_hold, to_continue)`` for a conflicting pair.

    The weaker aircraft is held. When the ordering keys tie exactly, the
    lexicographically greater id is held, which is a total and symmetric rule
    so the decision never depends on argument order.
    """
    if priority_key(first) == priority_key(second):
        if first.id < second.id:
            return second, first
        return first, second
    # sorted() is ascending and a smaller key means *stronger*, so the
    # first element is the aircraft that keeps flying.
    stronger, weaker = sorted((first, second), key=priority_key)
    return weaker, stronger


def yield_reason(
    to_hold: Aircraft, to_continue: Aircraft
) -> str:
    """Explain which rule made ``to_hold`` the weaker aircraft.

    Naming the deciding rule keeps the event log honest: a plain
    "class 1 outranks 1" would be true but would not tell a reader whether the
    priority class, the request time or the id decided it.
    """
    held_class = PRIORITY_CLASS[operation_for(to_hold)]
    kept_class = PRIORITY_CLASS[operation_for(to_continue)]
    if held_class != kept_class:
        kept_operation = operation_for(to_continue).name
        return (
            f"{kept_operation.lower()} outranks "
            f"{operation_for(to_hold).name.lower()}"
        )
    if to_hold.request_time != to_continue.request_time:
        return "later runway request yields to the earlier one"
    return "identical priority and request time, so the greater id yields"


def urgency_key(conflict: ConflictResult) -> tuple[float, str, str]:
    """Sort key placing the most urgent conflict first.

    Primary key is ``time_to_conflict`` (soonest first); the two sorted ids are
    tie-breakers so the ordering is total and deterministic. A result with no
    conflict has no urgency, so it sorts to the end with infinite distance.
    """
    if not conflict.conflict or conflict.time_to_conflict is None:
        return (float("inf"), "", "")
    low, high = sorted((conflict.aircraft_a.id, conflict.aircraft_b.id))
    return (conflict.time_to_conflict, low, high)


def hold(
    aircraft: Aircraft,
    current_tick: int,
    hold_duration: int = HOLD_DURATION,
) -> int:
    """Apply a HOLD to ``aircraft`` and return the new ``holding_until``.

    An existing hold is never shortened: a new instruction extends it to
    ``max(previous, current_tick + hold_duration)``, so repeated conflicts over
    consecutive ticks produce one long hold rather than a slipping deadline.
    This is what makes the per-conflict actions order-independent.
    """
    requested_until = current_tick + hold_duration
    new_until = (
        requested_until
        if aircraft.holding_until is None
        else max(aircraft.holding_until, requested_until)
    )
    aircraft.holding_until = new_until
    return new_until


def resolve(
    conflict: ConflictResult,
    current_tick: int,
    hold_duration: int = HOLD_DURATION,
) -> ResolutionResult:
    """Resolve one conflict with a HOLD and return a description of the action.

    A non-conflicting result yields :attr:`ResolutionAction.NO_ACTION` and
    touches nothing. Otherwise the lower-priority aircraft is held; the
    resolver writes only ``holding_until``.
    """
    if not conflict.conflict:
        return ResolutionResult(
            conflict=conflict,
            action=ResolutionAction.NO_ACTION,
            reason="no conflict predicted",
        )

    first, second = conflict.aircraft_a, conflict.aircraft_b
    to_hold, to_continue = choose_aircraft_to_hold(first, second)
    reason = (
        f"{to_hold.id} yields to {to_continue.id} "
        f"({yield_reason(to_hold, to_continue)}), "
        f"time to conflict {conflict.time_to_conflict}"
    )
    until = hold(to_hold, current_tick, hold_duration)
    return ResolutionResult(
        conflict=conflict,
        action=ResolutionAction.HOLD_LOWER_PRIORITY,
        reason=reason,
        held_aircraft=to_hold,
        holding_until=until,
    )


class ConflictResolver:
    """Resolve a whole tick's worth of predicted conflicts, most urgent first."""

    def __init__(self, hold_duration: int = HOLD_DURATION) -> None:
        if hold_duration <= 0:
            raise ValueError(
                f"hold_duration must be positive, got {hold_duration!r}"
            )
        self._hold_duration = hold_duration

    @property
    def hold_duration(self) -> int:
        """Length in ticks of the HOLD instructions this resolver issues."""
        return self._hold_duration

    def resolve_all(
        self,
        conflicts: list[ConflictResult],
        current_tick: int,
    ) -> list[ResolutionResult]:
        """Resolve ``conflicts`` and return one :class:`ResolutionResult` each.

        Results come back in urgency order, so the caller can log the most
        pressing instruction first. Only conflicting entries produce holds;
        non-conflicting results are still reported, as
        :attr:`ResolutionAction.NO_ACTION`.
        """
        return [
            resolve(conflict, current_tick, self._hold_duration)
            for conflict in sorted(conflicts, key=urgency_key)
        ]
