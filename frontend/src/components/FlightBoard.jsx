import { COL, PHASE } from '../interp'

// Lifecycle as the backend defines it; a bar fills as each aircraft advances.
const STEPS = ['AT_GATE', 'TAXIING_TO_RUNWAY', 'WAITING_FOR_RUNWAY', 'LINE_UP', 'TAKEOFF', 'CLIMB',
  'CRUISE', 'DESCENT', 'APPROACH', 'LANDING', 'TAXIING_TO_GATE', 'COMPLETED']

export default function FlightBoard({ frame, first }) {
  return (
    <table className="board">
      <thead><tr><th>Flight</th><th>Type</th><th>Now</th><th>Journey</th></tr></thead>
      <tbody>
        {frame.map((a, n) => {
          const at = STEPS.indexOf(a.s)
          return (
            <tr key={a.id}>
              <td>{a.id}</td>
              <td className="dim">{first[n].s === 'LANDING' ? 'Arrival' : 'Departure'}</td>
              <td style={{ color: COL[a.s] }}>{PHASE[a.s]}{a.s === 'COMPLETED' && ' ✓'}{a.hold && <em> · HOLD</em>}</td>
              <td><div className="steps" title={`Step ${at + 1} of ${STEPS.length}`}>
                {STEPS.map((s, k) => <i key={s} title={PHASE[s]} style={k <= at ? { background: COL[a.s] } : undefined} />)}
              </div></td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
