import { useCallback, useEffect, useRef, useState } from 'react'

export interface PlaybackController {
  time: number
  duration: number
  playing: boolean
  speed: number
  play: () => void
  pause: () => void
  restart: () => void
  seek: (time: number) => void
  setSpeed: (speed: number) => void
}

export function usePlayback(duration: number): PlaybackController {
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeedState] = useState(1)
  const lastFrame = useRef<number | null>(null)
  const lastUiUpdate = useRef(0)
  const pendingElapsed = useRef(0)

  useEffect(() => {
    if (!playing) {
      lastFrame.current = null
      pendingElapsed.current = 0
      return
    }
    let frame = 0
    const animate = (timestamp: number) => {
      if (lastFrame.current === null) lastFrame.current = timestamp
      const elapsed = ((timestamp - lastFrame.current) / 1000) * speed
      lastFrame.current = timestamp
      pendingElapsed.current += elapsed
      if (timestamp - lastUiUpdate.current >= 24) {
        lastUiUpdate.current = timestamp
        const elapsedSinceUpdate = pendingElapsed.current
        pendingElapsed.current = 0
        setTime((current) => {
          const next = Math.min(duration, current + elapsedSinceUpdate)
          if (next >= duration) setPlaying(false)
          return next
        })
      }
      frame = requestAnimationFrame(animate)
    }
    frame = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(frame)
  }, [duration, playing, speed])

  const play = useCallback(() => {
    setTime((current) => (current >= duration ? 0 : current))
    setPlaying(true)
  }, [duration])
  const pause = useCallback(() => setPlaying(false), [])
  const restart = useCallback(() => {
    setPlaying(false)
    setTime(0)
  }, [])
  const seek = useCallback(
    (next: number) => setTime(Math.max(0, Math.min(duration, next))),
    [duration],
  )
  const setSpeed = useCallback((next: number) => setSpeedState(next), [])

  return { time, duration, playing, speed, play, pause, restart, seek, setSpeed }
}
