import { motion, AnimatePresence } from 'framer-motion'
import { useEffect, useMemo, useState } from 'react'
import FlightBoard from './components/FlightBoard'
import Ground from './components/Ground'
import Radar from './components/Radar'
import { EVENT, plain } from './interp'
import { useClock } from './useClock'

const fade = (d) => ({ initial: { opacity: 0, y: 24 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.9, delay: d } })

export default function App() {
  const [data, setData] = useState(null)
  const [err, setErr] = useState(null)
  useEffect(() => {
    Promise.all(['network', 'replay'].map((p) => fetch(`/api/${p}`).then((r) => r.json())))
      .then(([n, r]) => setData({ ...n, ...r }))
      .catch((e) => setErr(String(e)))
  }, [])
  if (err) return <p className="err">Backend unreachable: {err}. Run: uvicorn backend.api:app</p>
  if (!data) return <p className="err" style={{ color: 'var(--cyan)' }}>Loading simulation…</p>
  return <Dashboard data={data} />
}

function Dashboard({ data }) {
  const { frames, events } = data
  const last = frames.length - 1
  const { t, playing, setPlaying, seek, speed, setSpeed, restart } = useClock(last)
  const tick = Math.floor(t)
  const f = frames[tick]
  const air = f.filter((a) => a.x !== undefined).length
  const done = f.filter((a) => a.s === 'COMPLETED').length
  const stats = [
    ['In the air', air, 'Aircraft currently flying the airspace pattern'],
    ['On the ground', f.length - air - done, 'Aircraft at a gate, taxiing, or queued for the runway'],
    ['Finished', done, 'Aircraft that completed their whole journey'],
    ['Holding', f.filter((a) => a.hold).length, 'Aircraft told to wait in the air so two planes never get too close'],
  ]
  const log = useMemo(() => events.filter((e) => e[0] <= tick).reverse(), [events, tick])
  const closed = useMemo(() => {
    const s = new Set()
    events.forEach(([tk, k, m]) => {
      if (tk > tick || (k !== 'BLOCK' && k !== 'UNBLOCK')) return
      const key = m.split(' ')[1].split('-').sort().join('-')
      if (k === 'BLOCK') s.add(key)
      else s.delete(key)
    })
    return s
  }, [events, tick])

  return (
    <>
      <div className="bgglow" />
      <header>
        <div className="hero">
        <div>
        <motion.div className="tag" {...fade(0)}>DAA PROJECT · AIRPORT TRAFFIC SIMULATOR</motion.div>
        <motion.h1 {...fade(0.15)}>TOWER</motion.h1>
        <motion.p {...fade(0.3)}>A tiny airport run entirely by algorithms. {f.length} aircraft leave their gates, share
          a single runway, fly a loop and land — and you are watching the real backend simulation, replayed one tick
          (one time step) at a time.</motion.p>
        <motion.div className="stats" {...fade(0.45)}>
          {stats.map(([l, v, tip]) => (
            <div className="stat" key={l} title={tip}>
              <AnimatePresence mode="popLayout"><motion.b key={v} initial={{ y: -14, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: 14, opacity: 0 }}>{v}</motion.b></AnimatePresence>
              <span>{l}</span>
            </div>
          ))}
        </motion.div>
        </div>
        <motion.ol className="guide" {...fade(0.6)}>
          <li><b>Ground map</b> — planes taxi along the shortest path found by <em>Dijkstra's algorithm</em>. A red dashed line is a closed taxiway that forces a reroute.</li>
          <li><b>Runway queue</b> — there is only one runway, so a <em>priority queue</em> decides who goes next. Landings beat take-offs.</li>
          <li><b>Radar</b> — in the air, the simulator predicts if two planes will get too close and orders one to <em>hold</em>.</li>
          <li><b>Flight board &amp; log</b> — every aircraft's progress and every decision, in plain words. Hover anything for details.</li>
        </motion.ol>
        </div>
      </header>
      <main>
        <section className="panel"><h2>GROUND MAP</h2>
          <p className="cap">The airport from above: gates around the edge, taxiways in the middle, runway entries inside. Coloured dots are aircraft.</p>
          <Ground edges={data.edges} frames={frames} t={t} closed={closed} /></section>
        <section className="panel"><h2>AIRSPACE RADAR</h2>
          <p className="cap">After take-off every aircraft flies the dashed loop through five waypoints, then comes back to land. Labels show altitude in feet.</p>
          <Radar waypoints={data.waypoints} frames={frames} t={t} /></section>
        <section className="panel"><h2>FLIGHT BOARD</h2>
          <p className="cap">Where each aircraft is in its journey: gate → taxi → runway → flight → landing → gate.</p>
          <FlightBoard frame={f} first={frames[0]} />
        </section>
        <section className="panel"><h2>EVENT LOG</h2>
          <p className="cap">Every decision the simulator made, newest first.</p>
          <div className="log">
          {!log.length && <div>Nothing yet — press play.</div>}
          <AnimatePresence initial={false}>
            {log.map((e) => (
              <motion.div key={e.join('|')} className={e[1]} initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }}>
                <i>tick {e[0]}</i><b title={e[1]}>{EVENT[e[1]] ?? e[1]}</b><span>{plain(e[2])}</span>
              </motion.div>
            ))}
          </AnimatePresence>
          </div>
        </section>
      </main>
      <div className="ctl">
        <button onClick={() => (t >= last ? restart() : setPlaying(!playing))}>{playing ? '❚❚ Pause' : t >= last ? '↺ Replay' : '▶ Play'}</button>
        <button onClick={restart}>⏮ Start over</button>
        <label className="scrub">
          <span>Drag to jump through time</span>
          <input type="range" min="0" max={last} step="0.01" value={t} onChange={(e) => seek(+e.target.value)} aria-label="Simulation time" />
        </label>
        <span className="clock">tick {tick} / {last}</span>
        <label className="speed">Speed <select value={speed} onChange={(e) => setSpeed(+e.target.value)}>{[1, 2, 4].map((s) => <option key={s} value={s}>{s}×</option>)}</select></label>
      </div>
    </>
  )
}
