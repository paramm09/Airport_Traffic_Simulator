export const COL = {
  AT_GATE: '#5b6f82', TAXIING_TO_RUNWAY: '#ffb347', WAITING_FOR_RUNWAY: '#ff4d6d', LINE_UP: '#ff4d6d',
  TAKEOFF: '#5dffa8', CLIMB: '#5dffa8', CRUISE: '#38e8ff', DESCENT: '#a78bfa', APPROACH: '#a78bfa',
  LANDING: '#a78bfa', TAXIING_TO_GATE: '#ffb347', COMPLETED: '#3d7a5c', HOLDING_NO_ROUTE: '#ff4d6d',
}
const lerp = (a, b, k) => a + (b - a) * k

// Smooth aircraft state between tick i and i+1.
export function at(frames, t) {
  const i = Math.min(Math.floor(t), frames.length - 1)
  const j = Math.min(i + 1, frames.length - 1)
  const k = t - i
  return frames[i].map((a, n) => {
    const b = frames[j][n]
    const o = { ...a }
    if (a.a !== undefined && b.a !== undefined) {
      if (a.a === b.a && a.b === b.b) o.t = lerp(a.t, b.t, k)
      else if (k > 0.5) Object.assign(o, { a: b.a, b: b.b, t: b.t })
    }
    if (a.x !== undefined && b.x !== undefined) {
      o.x = lerp(a.x, b.x, k); o.y = lerp(a.y, b.y, k); o.alt = lerp(a.alt, b.alt, k)
    }
    return o
  })
}

// Pair each aircraft with its position and an index among aircraft sharing that spot,
// so stacked aircraft (e.g. queued at a runway) get separate labels.
export function stack(list, pos) {
  const seen = {}
  return list.map((a) => {
    const [x, y] = pos(a), k = `${Math.round(x / 8)},${Math.round(y / 8)}`
    seen[k] = (seen[k] ?? -1) + 1
    return [a, x, y, seen[k]]
  })
}

// Plain-English names for the backend's lifecycle states, in lifecycle order.
export const PHASE = {
  AT_GATE: 'At gate', TAXIING_TO_RUNWAY: 'Taxiing to runway', WAITING_FOR_RUNWAY: 'Waiting for runway',
  LINE_UP: 'Lining up', TAKEOFF: 'Taking off', CLIMB: 'Climbing', CRUISE: 'Cruising', DESCENT: 'Descending',
  APPROACH: 'On approach', LANDING: 'Landing', TAXIING_TO_GATE: 'Taxiing in', COMPLETED: 'Done',
  HOLDING_NO_ROUTE: 'Stuck — no route',
}

// Plain-English labels for event-log kinds.
export const EVENT = {
  RUNWAY_REQUEST: 'Asks for runway', RUNWAY_ASSIGNED: 'Runway granted', RUNWAY_RELEASED: 'Runway freed',
  TAXI_COMPLETE: 'Taxi finished', AIR_LEG_COMPLETE: 'Waypoint reached', BLOCK: 'Taxiway closed',
  UNBLOCK: 'Taxiway reopened', BLOCKED: 'Path blocked', REROUTE: 'Rerouted', NO_ROUTE: 'No route',
  STILL_NO_ROUTE: 'Still no route', ROUTE_FOUND: 'Route found', HOLD: 'Hold ordered', HOLDING: 'Holding',
}

// Swap backend state names inside a log message for their plain-English phase.
const STATE_RE = new RegExp(`\\b(${Object.keys(PHASE).join('|')})\\b`, 'g')
export const plain = (msg) => msg.replace(STATE_RE, (s) => PHASE[s].toLowerCase())
