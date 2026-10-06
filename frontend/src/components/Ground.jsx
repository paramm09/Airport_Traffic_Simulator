import { useMemo, useState } from 'react'
import { at, stack } from '../interp'
import Plane from './Plane'

const NODE = { G: 'Gate', T: 'Taxiway junction', R: 'Runway entry' }
const cx = 450, cy = 350
const key = (a, b) => [a, b].sort().join('-')

function layout(edges) {
  const P = {}, near = {}
  for (let i = 1; i <= 10; i++) {
    const a = ((-90 + (i - 1) * 36) * Math.PI) / 180
    P['T' + i] = [cx + 220 * Math.cos(a), cy + 220 * Math.sin(a)]
    P['G' + i] = [cx + 310 * Math.cos(a), cy + 290 * Math.sin(a)]
  }
  edges.forEach(([a, b]) => { if (b[0] === 'R') (near[b] ||= []).push(P[a]) })
  for (const r in near) {
    const [p, q] = near[r]
    P[r] = [((p[0] + q[0]) / 2) * 0.55 + cx * 0.45, ((p[1] + q[1]) / 2) * 0.55 + cy * 0.45]
  }
  return P
}

export default function Ground({ edges, frames, t, closed }) {
  const P = useMemo(() => layout(edges), [edges])
  const [pick, setPick] = useState(null)
  const [route, setRoute] = useState(null)

  const click = async (n) => {
    if (!pick) { setPick(n); setRoute(null); return }
    const r = await fetch(`/api/route?start=${pick}&end=${n}`).then((x) => x.json())
    setRoute(r); setPick(null)
  }
  const onRoute = new Set()
  route?.path.forEach((n, i) => i && onRoute.add(key(route.path[i - 1], n)))

  return (
    <>
      <svg viewBox="0 0 900 700">
        {edges.map(([a, b]) => {
          const k = key(a, b), x = closed.has(k), r = onRoute.has(k)
          return <line key={k} x1={P[a][0]} y1={P[a][1]} x2={P[b][0]} y2={P[b][1]}
            stroke={x ? '#ff4d6d' : r ? '#5dffa8' : '#1d3040'} strokeWidth={r ? 3 : 1.5}
            strokeDasharray={x || r ? '6 6' : undefined} className={x || r ? 'dash' : undefined} />
        })}
        {Object.entries(P).map(([n, [x, y]]) => (
          <g key={n} className="node" onClick={() => click(n)}>
            <title>{`${NODE[n[0]]} ${n} — click to ${pick ? 'route here' : 'start a route'}`}</title>
            <circle cx={x} cy={y} r={n[0] === 'R' ? 9 : n[0] === 'G' ? 6 : 4}
              fill={n === pick ? '#fff' : n[0] === 'R' ? '#38e8ff' : n[0] === 'G' ? '#a78bfa' : '#2d4458'} />
            <text x={x + 10} y={y - 8}>{n}</text>
          </g>
        ))}
        {stack(at(frames, t).filter((a) => a.a !== undefined && a.s !== 'COMPLETED'), (a) => {
          const [x1, y1] = P[a.a], [x2, y2] = P[a.b]
          return [x1 + (x2 - x1) * a.t, y1 + (y2 - y1) * a.t]
        }).map(([a, x, y, i]) => <Plane key={a.id} x={x} y={y} i={i} state={a.s} label={a.id} hold={a.hold} />)}
      </svg>
      <ul className="legend">
        <li><i style={{ background: '#a78bfa' }} />Gate (G)</li>
        <li><i style={{ background: '#2d4458' }} />Taxiway junction (T)</li>
        <li><i style={{ background: '#38e8ff' }} />Runway entry (R)</li>
        <li><i className="bar" style={{ borderColor: '#ff4d6d' }} />Closed taxiway</li>
        <li><i className="bar" style={{ borderColor: '#5dffa8' }} />Your Dijkstra route</li>
      </ul>
      <div className="route">
        <b>Try Dijkstra:</b>{' '}
        {pick ? <>start <u>{pick}</u> selected — now click a destination node</>
          : route ? (route.path.length
            ? <>shortest path {route.path.join(' → ')} · total distance <u>{route.cost}</u></>
            : 'no path between those nodes')
          : 'click any node, then another, to see the shortest taxi path between them'}
        {(pick || route) && <button className="link" onClick={() => { setPick(null); setRoute(null) }}>clear</button>}
      </div>
    </>
  )
}
