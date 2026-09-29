"""Deterministic runway scheduling for the shared airport runway.

Several aircraft can be waiting for access to the same runway, and the
simulation must pick — deterministically — who goes next. That is exactly the
"minimum element of a dynamic set" problem, solved here with a **binary
min-heap** (``heapq``) over the lexicographic priority tuple::

    (priority_class, request_time, aircraft_id)

    priority_class   0 = LANDING, 1 = TAKEOFF   (0 outranks 1)
    request_time     earlier requests win within the same class
    aircraft_id      final deterministic tie-breaker

The DAA data structure is the binary min-heap / priority queue; this module is
only the scheduling policy wrapped around it. Responsibility split (kept
deliberately separate):

    Dijkstra           "what is the shortest taxi route?"
    taxi_step          "how does the aircraft move along that route?"
    RunwayScheduler    "which WAITING aircraft gets the shared runway next?"
    Aircraft           "which lifecycle state is the aircraft currently in?"

The runway is a binary resource: FREE or BUSY. Selection happens only when the
runway is FREE; the current occupant is never interrupted (a landing waiting in
the queue does NOT push an aircraft already on the runway). When the runway is
released at ``release_tick`` the next aircraft is selected.
"""

from __future__ import annotations

import enum
import heapq

from backend.models.aircraft import Aircraft, AircraftState

# One runway occupant keeps the runway for this many simulation ticks.
RUNWAY_OPERATION_TIME = 5


class RunwayOperation(enum.Enum):
    """The kind of runway operation an aircraft requests."""

    LANDING = enum.auto()
    TAKEOFF = enum.auto()


# Selection policy: LANDING (0) always outranks TAKEOFF (1). This only chooses
# the *next* occupant; an aircraft already on the runway is never interrupted.
PRIORITY_CLASS: dict[RunwayOperation, int] = {
    RunwayOperation.LANDING: 0,
    RunwayOperation.TAKEOFF: 1,
}


class RunwayScheduler:
    """Choose the next aircraft for one shared runway using a binary min-heap.

    Waiting requests are stored as lightweight ``(priority_class,
    request_time, aircraft_id)`` tuples inside a ``heapq`` min-heap, while the
    aircraft objects themselves live in a ``_waiting`` dictionary keyed by
    ``id`` — the heap never duplicates an entire aircraft.

    Complexity (n = number of aircraft waiting for the runway):

        insert (request)      O(log n)    heapq.heappush
        extract-min (select)  O(log n)    heapq.heappop
        peek (minimum)        O(1)        self._heap[0]
        space                 O(n)        one heap + one dict entry per waiter

    A heap is preferred over re-scanning the waiting list: scanning for the
    minimum costs O(n) on *every* selection, so a fresh linear pass each time is
    O(n²) over n selections, while the heap pays only O(log n) per insert and
    per extract.
    """

    def __init__(self, operation_time: int = RUNWAY_OPERATION_TIME) -> None:
        """Create an empty scheduler with the given runway occupation time."""
        if operation_time <= 0:
            raise ValueError(
                f"operation_time must be positive, got {operation_time!r}"
            )
        self._operation_time = operation_time
        self._heap: list[tuple[int, int, str]] = []
        self._waiting: dict[str, Aircraft] = {}
        self._occupant: Aircraft | None = None
        self._release_tick: int | None = None

    # ------------------------------------------------------------------
    # Requests and the waiting heap
    # ------------------------------------------------------------------
    def request(
        self,
        aircraft: Aircraft,
        operation: RunwayOperation,
        request_time: int,
    ) -> bool:
        """Queue ``aircraft`` for the runway if the request is valid.

        Returns:
            ``True`` when the aircraft is queued for the runway, ``False`` for a
            duplicate request (the aircraft is already waiting, or is currently
            occupying the runway) — the heap is left untouched in that case.

        Raises:
            ValueError: if ``aircraft`` is ``None``, ``operation`` is not a
                :class:`RunwayOperation`, or ``request_time`` is negative.

        On acceptance ``aircraft.request_time`` is set to ``request_time`` so a
        heap entry and its aircraft always agree. The operation type is passed
        explicitly because the scheduler must know whether a request is a
        landing or a takeoff, and the lifecycle state alone does not reliably
        say so; no extra aircraft state is needed.
        """
        if aircraft is None:
            raise ValueError("aircraft must not be None")
        if not isinstance(operation, RunwayOperation):
            raise ValueError(
                "operation must be a RunwayOperation, " f"got {operation!r}"
            )
        if request_time < 0:
            raise ValueError(
                f"request_time must not be negative, got {request_time!r}"
            )
        if aircraft.id in self._waiting:
            return False
        if self._occupant is not None and self._occupant.id == aircraft.id:
            return False

        aircraft.request_time = request_time
        self._waiting[aircraft.id] = aircraft
        heapq.heappush(
            self._heap,
            (PRIORITY_CLASS[operation], request_time, aircraft.id),
        )
        return True

    def waiting_count(self) -> int:
        """Number of aircraft currently queued for the runway."""
        return len(self._waiting)

    def peek(self) -> Aircraft | None:
        """Return the next runway candidate without selecting it, or ``None``.

        O(1): ``request`` rejects duplicates, so the heap never holds a stale
        top entry under the supported API and the minimum is exactly
        ``self._heap[0]``. (The lazy-deletion path is in ``select_next``.)
        """
        if not self._heap:
            return None
        return self._waiting.get(self._heap[0][2])

    # ------------------------------------------------------------------
    # Runway state
    # ------------------------------------------------------------------
    def is_busy(self) -> bool:
        """Return ``True`` while an aircraft occupies the runway."""
        return self._occupant is not None

    def is_free(self) -> bool:
        """Return ``True`` while no aircraft is on the runway."""
        return self._occupant is None

    def release_tick(self) -> int | None:
        """Tick at which the runway will become free, else ``None``.

        The value is only meaningful while the runway is BUSY.
        """
        return self._release_tick

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def select_next(self, current_tick: int) -> Aircraft | None:
        """Give the runway to the highest-priority waiting aircraft.

        Only legal while the runway is FREE: selecting while BUSY raises
        ``RuntimeError``, because the current occupant must finish its
        operation first (no preemption).

        Pops the minimum ``(priority_class, request_time, aircraft_id)`` entry,
        lazily skipping stale entries, marks the runway BUSY until
        ``current_tick + RUNWAY_OPERATION_TIME``, and returns the new occupant
        (``None`` if nobody is waiting). A selected aircraft that is in
        ``WAITING_FOR_RUNWAY`` moves to ``LINE_UP``; the scheduler deliberately
        does not perform ``LINE_UP -> TAKEOFF`` or the arrival pipeline.
        """
        if self.is_busy():
            raise RuntimeError(
                "cannot select a runway occupant while the runway is busy"
            )
        aircraft = self._pop_valid_entry()
        if aircraft is None:
            return None
        self._occupant = aircraft
        self._release_tick = current_tick + self._operation_time
        if aircraft.can_transition_to(AircraftState.LINE_UP):
            aircraft.transition_to(AircraftState.LINE_UP)
        return aircraft

    def step(self, current_tick: int) -> Aircraft | None:
        """Advance the scheduler to ``current_tick`` and select if possible.

        If the runway is BUSY and ``current_tick`` has reached
        ``release_tick``, the runway is released first; then, if the runway is
        FREE and aircraft are waiting, the next occupant is selected. Returns
        the newly selected aircraft, or ``None``.
        """
        if (
            self.is_busy()
            and self._release_tick is not None
            and current_tick >= self._release_tick
        ):
            self._occupant = None
            self._release_tick = None
        if not self.is_busy():
            return self.select_next(current_tick)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _pop_valid_entry(self) -> Aircraft | None:
        """Pop the minimum heap entry that still represents a waiting aircraft.

        A popped entry is *stale* when its ``aircraft_id`` is no longer in
        ``_waiting`` (the aircraft has already been selected / its entry was
        superseded) or when its stored ``request_time`` no longer matches the
        aircraft's current one. Stale entries are skipped without selection —
        the same lazy-deletion pattern Dijkstra uses, which matters because
        ``heapq`` cannot decrease a key once pushed.
        """
        while self._heap:
            _priority, request_time, aircraft_id = heapq.heappop(self._heap)
            aircraft = self._waiting.get(aircraft_id)
            if aircraft is None or aircraft.request_time != request_time:
                continue  # stale heap entry: skip (lazy deletion)
            del self._waiting[aircraft_id]
            return aircraft
        return None