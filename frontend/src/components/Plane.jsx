import { COL, PHASE } from '../interp'

export default function Plane({ x, y, i = 0, state, label, hold }) {
  const c = COL[state]
  return (
    <g transform={`translate(${x} ${y})`}>
      <title>{`${label} — ${PHASE[state]}${hold ? ' (holding to keep separation)' : ''}`}</title>
      {hold && <circle r="14" fill="none" stroke="#ff4d6d" strokeWidth="2" className="dash" strokeDasharray="4 4" />}
      <circle r={7 - Math.min(i, 3)} fill={c} style={{ filter: `drop-shadow(0 0 8px ${c})` }} />
      <text x="12" y={4 + i * 14} className="plane-label" style={{ fill: c }}>{label}</text>
    </g>
  )
}
