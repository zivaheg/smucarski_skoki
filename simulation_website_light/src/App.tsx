import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { loadAppData } from './data'
import { ControlPanel } from './controls/ControlPanel'
import { usePlayback } from './hooks/usePlayback'
import type { CameraPreset } from './scene/JumpScene'
import { defaultSliders } from './simulation/controls'
import { calculateMetrics } from './simulation/metrics'
import type { AppData, JumpMetrics, Vector } from './simulation/model-types'
import { simulateSSM, stateAtTime } from './simulation/ssm'

const JumpScene = lazy(() =>
  import('./scene/JumpScene').then((module) => ({ default: module.JumpScene })),
)

function format(value: number | null, digits = 2): string {
  if (value === null || !Number.isFinite(value)) return '—'
  return value.toFixed(digits)
}

function Delta({ value }: { value: number }) {
  const normalized = Math.abs(value) < 0.005 ? 0 : value
  const className = normalized > 0 ? 'positive' : normalized < 0 ? 'negative' : 'neutral'
  return (
    <span className={`metric-delta ${className}`}>
      {normalized > 0 ? '+' : ''}{normalized.toFixed(2)} m vs. baseline
    </span>
  )
}

function MetricCards({ metrics, currentState, time }: { metrics: JumpMetrics; currentState: Vector; time: number }) {
  return (
    <section className="metric-grid" aria-label="Simulation outcomes">
      <article className="metric-card metric-card--featured">
        <div className="metric-card__icon" aria-hidden="true">↗</div>
        <div>
          <span>Jump distance</span>
          <strong>{format(metrics.displayedDistance)} <small>m</small></strong>
          <Delta value={metrics.distanceDelta} />
        </div>
      </article>
      <article className="metric-card">
        <span>Landing point</span>
        <strong>{format(metrics.endpoint[0], 1)} <small>m</small></strong>
        <small>Lateral {format(metrics.endpoint[1], 2)} m</small>
      </article>
      <article className="metric-card">
        <span>Maximum height</span>
        <strong>{format(metrics.maxZ)} <small>m</small></strong>
        <small>Relative to takeoff</small>
      </article>
      <article className="metric-card">
        <span>Lateral deviation</span>
        <strong>{format(metrics.maxAbsY)} <small>m</small></strong>
        <small>Maximum across flight</small>
      </article>
      <article className="metric-card">
        <span>Landing clearance</span>
        <strong>{format(metrics.landingClearance)} <small>m</small></strong>
        <small>Above hill profile</small>
      </article>
      <article className="metric-card">
        <span>Live speed · {time.toFixed(2)} s</span>
        <strong>{format(currentState[6]!, 1)} <small>m/s</small></strong>
        <small>X {format(currentState[0]!, 1)} · Z {format(currentState[2]!, 1)}</small>
      </article>
    </section>
  )
}

function Dashboard({ data }: { data: AppData }) {
  const { model, baseline, hill } = data
  const [sliders, setSliders] = useState(() => defaultSliders(model))
  const [showBaseline, setShowBaseline] = useState(true)
  const [showWind, setShowWind] = useState(true)
  const [showAxes, setShowAxes] = useState(false)
  const [showLabels, setShowLabels] = useState(true)
  const [controlsOpen, setControlsOpen] = useState(false)
  const [cameraCommand, setCameraCommand] = useState<{ preset: CameraPreset; nonce: number }>({
    preset: 'perspective',
    nonce: 0,
  })

  const baselineResult = useMemo(
    () => simulateSSM(model, baseline, defaultSliders(model)),
    [baseline, model],
  )
  const result = useMemo(
    () => simulateSSM(model, baseline, sliders),
    [baseline, model, sliders],
  )
  const metrics = useMemo(
    () => calculateMetrics(result, hill, baseline.referenceSimulation.displayedDistance),
    [baseline.referenceSimulation.displayedDistance, hill, result],
  )
  const duration = result.times[result.times.length - 1] ?? 0
  const playback = usePlayback(duration)
  const currentState = stateAtTime(result, playback.time)

  const setCamera = (preset: CameraPreset) =>
    setCameraCommand((current) => ({ preset, nonce: current.nonce + 1 }))

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true"><span /></div>
          <div>
            <strong>PLANICA</strong>
            <small>Performance Lab</small>
          </div>
        </div>
        <div className="topbar__meta">
          <span className="event-label">Ski jumping · Flight simulator</span>
          <div className="model-status"><i /> Model ready</div>
        </div>
      </header>

      <main className="main-content">
        <section className="page-heading">
          <div>
            <p className="eyebrow">Interactive performance model</p>
            <h1>Explore every metre of the flight.</h1>
            <p className="page-heading__copy">
              Adjust wind and athlete configuration, then compare the predicted jump with the average profile.
            </p>
          </div>
          <button className="button button--primary mobile-only" onClick={() => setControlsOpen(true)}>
            Tune the jump
          </button>
        </section>

        <MetricCards metrics={metrics} currentState={currentState} time={playback.time} />

        <section className="workspace">
          <div className="scene-column">
            <section className="scene-card" aria-labelledby="scene-title">
              <div className="scene-toolbar">
                <div>
                  <p className="eyebrow">Live trajectory</p>
                  <h2 id="scene-title">Predicted flight path</h2>
                </div>
                <div className="toolbar-cluster camera-presets" aria-label="Camera presets">
                  {(['perspective', 'side', 'top', 'landing'] as CameraPreset[]).map((preset) => (
                    <button
                      key={preset}
                      className={cameraCommand.preset === preset ? 'chip is-active' : 'chip'}
                      onClick={() => setCamera(preset)}
                    >
                      {preset}
                    </button>
                  ))}
                </div>
                <div className="toolbar-cluster scene-toggles">
                  <label><input type="checkbox" checked={showBaseline} onChange={(event) => setShowBaseline(event.target.checked)} /> Baseline</label>
                  <label><input type="checkbox" checked={showWind} onChange={(event) => setShowWind(event.target.checked)} /> Wind</label>
                  <label><input type="checkbox" checked={showLabels} onChange={(event) => setShowLabels(event.target.checked)} /> Markers</label>
                  <label><input type="checkbox" checked={showAxes} onChange={(event) => setShowAxes(event.target.checked)} /> Axes</label>
                </div>
              </div>
              <div className="scene-viewport">
                <Suspense fallback={<div className="module-loader">Preparing the hill…</div>}>
                  <JumpScene
                    baselineResult={baselineResult}
                    result={result}
                    hill={hill}
                    time={playback.time}
                    showBaseline={showBaseline}
                    showAxes={showAxes}
                    showWind={showWind}
                    showLabels={showLabels}
                    cameraCommand={cameraCommand}
                  />
                </Suspense>
                <div className="scene-legend">
                  <span><i className="legend-line legend-line--travelled" /> Current flight</span>
                  <span><i className="legend-line legend-line--predicted" /> Prediction</span>
                  {showBaseline && <span><i className="legend-line legend-line--baseline" /> Average</span>}
                  <small>Drag to rotate · scroll to zoom</small>
                </div>
                <div className="axis-key">
                  <span className="axis-x">X forward</span>
                  <span className="axis-y">Z vertical</span>
                  <span className="axis-z">Y lateral</span>
                </div>
              </div>
              <div className="scene-playback-mobile">
                <button className="button button--primary" onClick={playback.playing ? playback.pause : playback.play}>
                  {playback.playing ? 'Pause' : 'Play jump'}
                </button>
                <button className="button" onClick={playback.restart}>Restart</button>
                <input
                  type="range"
                  min={0}
                  max={duration}
                  step={0.01}
                  value={playback.time}
                  style={{ '--slider-progress': `${duration > 0 ? (playback.time / duration) * 100 : 0}%` } as React.CSSProperties}
                  onChange={(event) => playback.seek(Number(event.target.value))}
                  aria-label="Scene timeline"
                />
                <output>{playback.time.toFixed(2)} s</output>
              </div>
            </section>

            <div className="context-strip">
              <span className="context-strip__number">139</span>
              <span>flight states</span>
              <i />
              <span className="context-strip__number">{model.sampleIntervalSeconds.toFixed(2)} s</span>
              <span>sample interval</span>
            </div>
          </div>

          <div className={`control-drawer${controlsOpen ? ' is-open' : ''}`}>
            <ControlPanel
              model={model}
              sliders={sliders}
              onSlidersChange={setSliders}
              playback={playback}
              onClose={() => setControlsOpen(false)}
            />
          </div>
        </section>

        {controlsOpen && (
          <button className="drawer-scrim" onClick={() => setControlsOpen(false)} aria-label="Close controls" />
        )}

        <footer className="app-footer">
          <div>
            <strong>{model.modelName}</strong>
            <span>Predictive model · {model.training.acceptedSequences} accepted sequences</span>
          </div>
        </footer>
      </main>
    </div>
  )
}

export default function App() {
  const [data, setData] = useState<AppData | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    loadAppData()
      .then((loaded) => {
        if (active) setData(loaded)
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : String(reason))
      })
    return () => {
      active = false
    }
  }, [])

  if (error) {
    return (
      <main className="load-screen load-screen--error">
        <p className="eyebrow">Data loading failed</p>
        <h1>Could not start the simulation</h1>
        <pre>{error}</pre>
      </main>
    )
  }
  if (!data) {
    return (
      <main className="load-screen">
        <div className="loader" />
        <p className="eyebrow">Preparing flight data</p>
        <h1>Planica Performance Lab</h1>
      </main>
    )
  }
  return <Dashboard data={data} />
}
