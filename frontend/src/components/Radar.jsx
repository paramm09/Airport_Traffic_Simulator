import { at, stack } from '../interp'
import Plane from './Plane'

// waypoint plane is x 0..20, y 0..10; centre it on the scope at (320, 350)
const X = (x) => 80 + x * 24, Y = (y) => 470 - y * 24

const LEGEND = [['#ffb347', 'Taxiing'], ['#ff4d6d', 'Waiting / line-up'], ['#5dffa8', 'Departing'],
  ['#38e8ff', 'Cruise'], ['#a78bfa', 'Arriving'], ['#3d7a5c', 'Finished']]

export default function Radar({ waypoints, frames, t }) {
  return (
    <>
    <svg viewBox="0 0 640 700">
      <defs>
        <linearGradient id="sw" x1="0" x2="1"><stop offset="0" stopColor="#38e8ff" stopOpacity="0" /><stop offset="1" stopColor="#38e8ff" stopOpacity=".35" /></linearGradient>
      </defs>
      {[80, 135, 190, 245, 300].map((r) => <circle key={r} cx="320" cy="350" r={r} fill="none" stroke="#16222e" />)}
      <path d="M20 350H620M320 40V660" stroke="#16222e" />
      <path className="sweep" d="M320 350 L620 350 A300 300 0 0 0 596 234 Z" fill="url(#sw)" transform="rotate(-20 320 350)" />
      <polyline points={waypoints.map(([, x, y]) => `${X(x)},${Y(y)}`).join(' ')} fill="none" stroke="#2d4458" strokeDasharray="4 6" />
      {waypoints.map(([id, x, y, alt]) => (
        <g key={id}><title>{`Waypoint ${id}: aircraft fly here and level at ${alt} ft`}</title><rect x={X(x) - 3} y={Y(y) - 3} width="6" height="6" fill="#38e8ff" /><text x={X(x)} y={Y(y) - 10} textAnchor="middle">{id} · {alt}ft</text></g>
      ))}
      {stack(at(frames, t).filter((a) => a.x !== undefined), (a) => [X(a.x), Y(a.y)]).map(([a, x, y, i]) => (
        <Plane key={a.id} x={x} y={y} i={i} state={a.s} label={`${a.id} ${Math.round(a.alt)}ft`} hold={a.hold} />
      ))}
    </svg>
    <ul className="legend">
      {LEGEND.map(([c, l]) => <li key={l}><i style={{ background: c, boxShadow: `0 0 8px ${c}` }} />{l}</li>)}
      <li><i className="ring" />HOLD</li>
    </ul>
    </>
  )
}
