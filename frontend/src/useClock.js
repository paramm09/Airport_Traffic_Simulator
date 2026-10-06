import { useEffect, useRef, useState } from 'react'

// Fractional tick counter driven by requestAnimationFrame (3 ticks/sec at 1×).
export function useClock(last) {
  const [t, setT] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(2)
  const ref = useRef(0)
  useEffect(() => {
    if (!playing) return
    let id, prev = performance.now()
    const loop = (now) => {
      ref.current = Math.min(last, ref.current + (Math.max(0, now - prev) / 1000) * 3 * speed)
      prev = now
      setT(ref.current)
      if (ref.current >= last) return setPlaying(false)
      id = requestAnimationFrame(loop)
    }
    id = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(id)
  }, [playing, speed, last])
  const seek = (v) => { ref.current = v; setT(v) }
  const restart = () => { seek(0); setPlaying(true) }
  return { t, playing, setPlaying, seek, speed, setSpeed, restart }
}
