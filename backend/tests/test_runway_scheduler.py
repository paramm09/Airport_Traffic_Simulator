"""Phase 2C tests: the runway scheduler and its min-heap priority queue.

The scheduler answers *"which waiting aircraft gets the shared runway next?"*
using a binary min-heap over the lexicographic priority tuple::

    (priority_class, request_time, aircraft_id)

with ``priority_class`` 0 = LANDING and 1 = TAKEOFF. Landings outrank takeoffs
for the *next* occupant; earlier ``request_time`` wins within a class; and
``aircraft_id`` is the final deterministic tie-breaker. The runway is FREE or
BUSY, and the current occupant is never interrupted.
"""

import pytest

from backend.models.aircraft import Aircraft, AircraftState
from backend.simulation.runway_scheduler import (
    PRIORITY_CLASS,
    RUNWAY_OPERATION_TIME,
    RunwayOperation,
    RunwayScheduler,
)


def aircraft(aircraft_id: str, operation: RunwayOperation) -> Aircraft:
    """An aircraft in the lifecycle state that matches its runway operation.

    A departure has taxied to the runway (``WAITING_FOR_RUNWAY``); an arrival is
    on approach (``APPROACH``) and its runway entry is integrated in a later
    phase. The scheduler itself only needs the operation type and the id.
    """
    state = (
        AircraftState.WAITING_FOR_RUNWAY
        if operation is RunwayOperation.TAKEOFF
        else AircraftState.APPROACH
    )
    return Aircraft(id=aircraft_id, state=state)


def request_fleet(
    scheduler: RunwayScheduler, plan: list[tuple[str, RunwayOperation, int]]
) -> None:
    """Queue every ``(aircraft_id, operation, request_time)`` triple."""
    for aircraft_id, operation, request_time in plan:
        ac = aircraft(aircraft_id, operation)
        assert scheduler.request(ac, operation, request_time) is True


# ----------------------------------------------------------------------
# 1. Empty scheduler
# ----------------------------------------------------------------------
def test_empty_scheduler_is_free_and_selects_nothing():
    scheduler = RunwayScheduler()
    assert scheduler.is_free()
    assert not scheduler.is_busy()
    assert scheduler.waiting_count() == 0
    assert scheduler.peek() is None
    assert scheduler.release_tick() is None
    assert scheduler.select_next(0) is None
    assert scheduler.step(0) is None
    assert scheduler.is_free()  # a no-op selection must not make it busy


def test_runway_operation_time_constant():
    assert RUNWAY_OPERATION_TIME == 5


def test_operation_time_must_be_positive():
    with pytest.raises(ValueError):
        RunwayScheduler(operation_time=0)
    with pytest.raises(ValueError):
        RunwayScheduler(operation_time=-1)


def test_priority_class_mapping():
    assert PRIORITY_CLASS[RunwayOperation.LANDING] == 0
    assert PRIORITY_CLASS[RunwayOperation.TAKEOFF] == 1


# ----------------------------------------------------------------------
# 2. One takeoff request is selected
# ----------------------------------------------------------------------
def test_single_takeoff_request_is_selected():
    scheduler = RunwayScheduler()
    ac = Aircraft(id="AC001", state=AircraftState.WAITING_FOR_RUNWAY)
    assert scheduler.request(ac, RunwayOperation.TAKEOFF, 0) is True

    assert scheduler.waiting_count() == 1
    assert scheduler.peek() is ac
    assert scheduler.step(0) is ac
    assert scheduler.is_busy()
    assert scheduler.release_tick() == RUNWAY_OPERATION_TIME


# ----------------------------------------------------------------------
# 3. One landing request is selected
# ----------------------------------------------------------------------
def test_single_landing_request_is_selected():
    scheduler = RunwayScheduler()
    ac = Aircraft(id="AC001", state=AircraftState.APPROACH)
    assert scheduler.request(ac, RunwayOperation.LANDING, 0) is True
    assert scheduler.step(0) is ac
    assert scheduler.is_busy()


def test_request_sets_aircraft_request_time():
    scheduler = RunwayScheduler()
    ac = Aircraft(id="AC001", state=AircraftState.WAITING_FOR_RUNWAY)
    scheduler.request(ac, RunwayOperation.TAKEOFF, 7)
    assert ac.request_time == 7


# ----------------------------------------------------------------------
# 4.-6. Priority policy: landing > takeoff, earlier > later, id breaks ties
# ----------------------------------------------------------------------
def test_landing_beats_takeoff_for_the_next_occupant():
    scheduler = RunwayScheduler()
    request_fleet(scheduler, [
        ("AC001", RunwayOperation.TAKEOFF, 0),
        ("AC002", RunwayOperation.TAKEOFF, 0),
        ("AC003", RunwayOperation.LANDING, 1),
    ])
    assert scheduler.step(0).id == "AC003"


def test_earlier_landing_beats_later_landing():
    scheduler = RunwayScheduler()
    request_fleet(scheduler, [
        ("AC004", RunwayOperation.LANDING, 4),
        ("AC005", RunwayOperation.LANDING, 7),
    ])
    assert scheduler.step(0).id == "AC004"
    assert scheduler.step(5).id == "AC005"


def test_earlier_takeoff_beats_later_takeoff():
    scheduler = RunwayScheduler()
    request_fleet(scheduler, [
        ("AC002", RunwayOperation.TAKEOFF, 2),
        ("AC001", RunwayOperation.TAKEOFF, 0),
    ])
    assert scheduler.step(0).id == "AC001"
    assert scheduler.step(5).id == "AC002"


def test_aircraft_id_breaks_a_full_tie():
    scheduler = RunwayScheduler()
    request_fleet(scheduler, [
        ("AC002", RunwayOperation.TAKEOFF, 0),
        ("AC001", RunwayOperation.TAKEOFF, 0),
    ])
    assert scheduler.step(0).id == "AC001"

    scheduler = RunwayScheduler()
    request_fleet(scheduler, [
        ("AC002", RunwayOperation.LANDING, 0),
        ("AC001", RunwayOperation.LANDING, 0),
    ])
    assert scheduler.step(0).id == "AC001"


def test_spec_scenario_full_priority_order():
    # The scenario from the Phase 2C brief: AC001/AC002 takeoff at tick 0,
    # then AC003 landing at tick 1, then AC004/AC005 landing at ticks 4/7.
    scheduler = RunwayScheduler()
    request_fleet(scheduler, [
        ("AC001", RunwayOperation.TAKEOFF, 0),
        ("AC002", RunwayOperation.TAKEOFF, 0),
        ("AC003", RunwayOperation.LANDING, 1),
        ("AC004", RunwayOperation.LANDING, 4),
        ("AC005", RunwayOperation.LANDING, 7),
    ])
    assert scheduler.waiting_count() == 5

    assert scheduler.step(0).id == "AC003"   # landing outranks both takeoffs
    assert scheduler.step(5).id == "AC004"   # earlier landing beats AC005
    assert scheduler.step(10).id == "AC005"
    assert scheduler.step(15).id == "AC001"  # both takeoffs at t0, id wins
    assert scheduler.step(20).id == "AC002"
    assert scheduler.step(25) is None
    assert scheduler.is_free()
    assert scheduler.waiting_count() == 0


# ----------------------------------------------------------------------
# 7.-10. Runway busy/free behaviour
# ----------------------------------------------------------------------
def test_runway_becomes_busy_after_selection():
    scheduler = RunwayScheduler()
    ac = aircraft("AC001", RunwayOperation.TAKEOFF)
    scheduler.request(ac, RunwayOperation.TAKEOFF, 0)
    assert scheduler.is_free()
    scheduler.step(0)
    assert scheduler.is_busy()
    assert scheduler.release_tick() == RUNWAY_OPERATION_TIME


def test_second_aircraft_cannot_occupy_while_busy():
    scheduler = RunwayScheduler()
    ac1 = aircraft("AC001", RunwayOperation.TAKEOFF)
    ac2 = aircraft("AC002", RunwayOperation.TAKEOFF)
    scheduler.request(ac1, RunwayOperation.TAKEOFF, 0)
    scheduler.request(ac2, RunwayOperation.TAKEOFF, 1)
    assert scheduler.step(0) is ac1
    assert scheduler.is_busy()

    # While busy no selection happens and the queue is unaffected.
    assert scheduler.step(3) is None
    assert scheduler.is_busy()
    assert scheduler.waiting_count() == 1
    assert scheduler.peek() is ac2

    # Selecting directly while busy is a programming error, never a second
    # occupant.
    with pytest.raises(RuntimeError):
        scheduler.select_next(3)
    assert scheduler.is_busy()
    assert scheduler.release_tick() == 5


def test_multiple_aircraft_remain_queued_while_busy():
    scheduler = RunwayScheduler()
    request_fleet(scheduler, [
        ("AC001", RunwayOperation.TAKEOFF, 0),
        ("AC002", RunwayOperation.TAKEOFF, 0),
        ("AC003", RunwayOperation.TAKEOFF, 0),
    ])
    scheduler.step(0)  # AC001 takes the runway
    assert scheduler.waiting_count() == 2
    scheduler.step(1)
    scheduler.step(4)
    assert scheduler.waiting_count() == 2  # no selection while busy
    scheduler.step(5)                     # release, then AC002
    assert scheduler.waiting_count() == 1
    scheduler.step(10)                    # release, then AC003
    assert scheduler.waiting_count() == 0


def test_runway_releases_exactly_at_release_tick():
    scheduler = RunwayScheduler()
    ac = aircraft("AC001", RunwayOperation.TAKEOFF)
    scheduler.request(ac, RunwayOperation.TAKEOFF, 0)
    scheduler.step(0)
    assert scheduler.release_tick() == 5

    assert scheduler.step(4) is None
    assert scheduler.is_busy()
    assert scheduler.release_tick() == 5

    # Nothing is waiting, so after the release the runway sits FREE.
    assert scheduler.step(5) is None
    assert scheduler.is_free()
    assert scheduler.release_tick() is None


def test_next_aircraft_is_selected_after_release():
    scheduler = RunwayScheduler()
    ac1 = aircraft("AC001", RunwayOperation.TAKEOFF)
    ac2 = aircraft("AC002", RunwayOperation.TAKEOFF)
    scheduler.request(ac1, RunwayOperation.TAKEOFF, 0)
    scheduler.request(ac2, RunwayOperation.TAKEOFF, 1)
    assert scheduler.step(0) is ac1
    assert scheduler.step(5) is ac2
    assert scheduler.is_busy()
    assert scheduler.release_tick() == 10


def test_peek_returns_the_next_occupant_without_selecting():
    scheduler = RunwayScheduler()
    request_fleet(scheduler, [
        ("AC001", RunwayOperation.TAKEOFF, 0),
        ("AC002", RunwayOperation.TAKEOFF, 0),
        ("AC003", RunwayOperation.LANDING, 1),
    ])
    assert scheduler.peek().id == "AC003"
    assert scheduler.waiting_count() == 3  # peek is not a selection
    assert scheduler.step(0).id == "AC003"  # still there when selected


# ----------------------------------------------------------------------
# 13.-14. Duplicate / invalid requests
# ----------------------------------------------------------------------
def test_duplicate_request_is_rejected_while_waiting():
    scheduler = RunwayScheduler()
    ac = aircraft("AC001", RunwayOperation.TAKEOFF)
    assert scheduler.request(ac, RunwayOperation.TAKEOFF, 2) is True
    assert scheduler.request(ac, RunwayOperation.TAKEOFF, 99) is False
    assert ac.request_time == 2  # the duplicate must not touch the request
    assert scheduler.waiting_count() == 1
    assert scheduler.step(0) is ac  # selected exactly once
    assert scheduler.waiting_count() == 0


def test_duplicate_request_does_not_cause_double_occupancy():
    scheduler = RunwayScheduler()
    ac = aircraft("AC001", RunwayOperation.TAKEOFF)
    assert scheduler.request(ac, RunwayOperation.TAKEOFF, 0) is True
    assert scheduler.step(0) is ac
    # While AC001 occupies the runway, another request for the same aircraft
    # is rejected, so it can never occupy the runway twice.
    assert scheduler.request(ac, RunwayOperation.TAKEOFF, 10) is False
    assert scheduler.step(5) is None
    assert scheduler.is_free()
    assert scheduler.waiting_count() == 0


def test_invalid_requests_are_rejected_safely():
    scheduler = RunwayScheduler()
    ac = aircraft("AC001", RunwayOperation.TAKEOFF)
    with pytest.raises(ValueError):
        scheduler.request(ac, RunwayOperation.TAKEOFF, -1)
    with pytest.raises(ValueError):
        scheduler.request(ac, "TAKEOFF", 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        scheduler.request(None, RunwayOperation.TAKEOFF, 0)  # type: ignore[arg-type]
    assert scheduler.waiting_count() == 0
    assert scheduler.is_free()


# ----------------------------------------------------------------------
# 15. Aircraft state integration
# ----------------------------------------------------------------------
def test_selected_waiting_aircraft_transitions_to_line_up():
    scheduler = RunwayScheduler()
    ac = Aircraft(id="AC001", state=AircraftState.WAITING_FOR_RUNWAY)
    scheduler.request(ac, RunwayOperation.TAKEOFF, 0)
    assert scheduler.step(0) is ac
    assert ac.state is AircraftState.LINE_UP


def test_selected_landing_aircraft_state_is_left_unchanged():
    # The arrival -> LANDING pipeline is integrated in a later phase; Phase 2C
    # must not invent an illegal APPROACH -> LINE_UP transition.
    scheduler = RunwayScheduler()
    ac = Aircraft(id="AC001", state=AircraftState.APPROACH)
    scheduler.request(ac, RunwayOperation.LANDING, 0)
    assert scheduler.step(0) is ac
    assert ac.state is AircraftState.APPROACH


# ----------------------------------------------------------------------
# Cross-check: the heap order must match naive lexicographic sorting
# ----------------------------------------------------------------------
def test_selection_order_matches_the_lexicographic_key_sorting():
    plan = [
        ("AC010", RunwayOperation.TAKEOFF, 3),
        ("AC002", RunwayOperation.LANDING, 5),
        ("AC007", RunwayOperation.LANDING, 2),
        ("AC001", RunwayOperation.TAKEOFF, 0),
        ("AC003", RunwayOperation.TAKEOFF, 0),
        ("AC004", RunwayOperation.TAKEOFF, 0),
        ("AC005", RunwayOperation.LANDING, 1),
    ]
    expected = [
        aircraft_id
        for aircraft_id, operation, tick in sorted(
            plan, key=lambda item: (PRIORITY_CLASS[item[1]], item[2], item[0])
        )
    ]

    scheduler = RunwayScheduler()
    request_fleet(scheduler, plan)

    actual: list[str] = []
    tick = 0
    while scheduler.waiting_count() > 0:
        selected = scheduler.step(tick)
        assert selected is not None
        actual.append(selected.id)
        tick += RUNWAY_OPERATION_TIME
    assert actual == expected