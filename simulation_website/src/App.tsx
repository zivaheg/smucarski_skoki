import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { loadAppData } from './data'
import { ControlPanel } from './controls/ControlPanel'
import { usePlayback } from './hooks/usePlayback'
import type { CameraPreset } from './scene/JumpScene'
import { defaultSliders } from './simulation/controls'
import { calculateMetrics } from './simulation/metrics'
import type { AppData, JumpMetrics, Vector } from './simulation/model-types'
import { calculateSensitivity } from './simulation/sensitivity'
import { simulateSSM, stateAtTime } from './simulation/ssm'

const JumpScene = lazy(() =>
  import('./scene/JumpScene').then((module) => ({ default: module.JumpScene })),
)
const PlotDashboard = lazy(() =>
  import('./plots/PlotDashboard').then((module) => ({ default: module.PlotDashboard })),
)

function format(value: number | null, digits = 2): string {
  if (value === null || !Number.isFinite(value)) return '—'
  return value.toFixed(digits)
}

function Delta({ value }: { value: number }) {
  const normalized = Math.abs(value) < 0.005 ? 0 : value
  const className = normalized > 0 ? 'positive' : normalized < 0 ? 'negative' : 'neutral'
  return <span className={`metric-delta ${className}`}>{normalized > 0 ? '+' : ''}{normalized.toFixed(2)} m</span>
}

function MetricCards({ metrics, currentState, time }: { metrics: JumpMetrics; currentState: Vector; time: number }) {
  return (
    <div className="metric-grid" aria-label="Simulation outcomes">
      <article className="metric-card metric-card--featured">
        <span>Displayed distance</span>
        <strong>{format(metrics.displayedDistance)} <small>m</small></strong>
        <Delta value={metrics.distanceDelta} />
        <small>X/Z range · Y is lateral</small>
      </article>
      <article className="metric-card">
        <span>Landing X / Y</span>
        <strong>{format(metrics.endpoint[0], 1)} / {format(metrics.endpoint[1], 2)}</strong>
        <small>metres</small>
      </article>
      <article className="metric-card">
        <span>Maximum Z</span>
        <strong>{format(metrics.maxZ)} <small>m</small></strong>
        <small>relative to takeoff</small>
      </article>
      <article className="metric-card">
        <span>Max lateral deviation</span>
        <strong>{format(metrics.maxAbsY)} <small>m</small></strong>
        <small>|Y| across the flight</small>
      </article>
      <article className="metric-card">
        <span>Landing clearance</span>
        <strong>{format(metrics.landingClearance)} <small>m</small></strong>
        <small>against model hill profile</small>
      </article>
      <article className="metric-card">
        <span>Live state · {time.toFixed(2)} s</span>
        <strong>{format(currentState[6]!, 1)} <small>m/s</small></strong>
        <small>X {format(currentState[0]!, 1)} · Z {format(currentState[2]!, 1)}</small>
      </article>
    </div>
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
  const baselineDistance = baseline.referenceSimulation.displayedDistance
  const metrics = useMemo(
    () => calculateMetrics(result, hill, baselineDistance),
    [baselineDistance, hill, result],
  )
  const sensitivity = useMemo(
    () => calculateSensitivity(model, baseline, hill, sliders),
    [baseline, hill, model, sliders],
  )
  const duration = result.times[result.times.length - 1] ?? 0
  const playback = usePlayback(duration)
  const currentState = stateAtTime(result, playback.time)

  const setCamera = (preset: CameraPreset) =>
    setCameraCommand((current) => ({ preset, nonce: current.nonce + 1 }))

  return (
    <div className="app-shell">
      <div className={`control-drawer${controlsOpen ? ' is-open' : ''}`}>
        <ControlPanel
          model={model}
          sliders={sliders}
          onSlidersChange={setSliders}
          playback={playback}
          onClose={() => setControlsOpen(false)}
        />
      </div>
      {controlsOpen && <button className="drawer-scrim" onClick={() => setControlsOpen(false)} aria-label="Close controls" />}

      <main className="main-content">
        <header className="topbar">
          <button className="button mobile-only" onClick={() => setControlsOpen(true)}>Controls</button>
          <div className="brand-mark" aria-hidden="true"><span /></div>
          <div className="topbar__title">
            <p className="eyebrow">Planica dynamics explorer</p>
            <h1>Ski Jump <em>SSM Lab</em></h1>
          </div>
          <div className="model-status">
            <i /> Model loaded
            <small>{model.trainedHorizon} states · Δt {model.sampleIntervalSeconds}s</small>
          </div>
        </header>

        <section className="intro-strip">
          <p>
            Explore model-predicted responses to wind and jumper configuration. Results describe this fitted linear SSM—not physical causality.
          </p>
          <div className="model-badges">
            <span>A {model.matrices.A.length}×{model.matrices.A[0]!.length}</span>
            <span>B {model.matrices.B.length}×{model.matrices.B[0]!.length}</span>
            <span>ρ(A) {model.diagnostics.spectralRadiusA.toFixed(4)}</span>
          </div>
        </section>

        <MetricCards metrics={metrics} currentState={currentState} time={playback.time} />

        <section className="scene-card" aria-labelledby="scene-title">
          <div className="scene-toolbar">
            <div>
              <p className="eyebrow">Interactive terrain</p>
              <h2 id="scene-title">Predicted flight path</h2>
            </div>
            <div className="toolbar-cluster" aria-label="Camera presets">
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
              <label title="Show wind direction and strength in three hill zones"><input type="checkbox" checked={showWind} onChange={(event) => setShowWind(event.target.checked)} /> Wind arrows</label>
              <label><input type="checkbox" checked={showLabels} onChange={(event) => setShowLabels(event.target.checked)} /> Labels</label>
              <label><input type="checkbox" checked={showAxes} onChange={(event) => setShowAxes(event.target.checked)} /> Axes</label>
            </div>
          </div>
          <div className="scene-viewport">
            <Suspense fallback={<div className="module-loader">Preparing 3D terrain…</div>}>
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
              <span><i className="legend-line legend-line--travelled" /> Travelled path</span>
              <span><i className="legend-line legend-line--predicted" /> Predicted guide</span>
              {showBaseline && <span><i className="legend-line legend-line--baseline" /> Baseline</span>}
              <small>drag to rotate · scroll to zoom · right-drag to pan</small>
            </div>
            <div className="axis-key">
              <span className="axis-x">X forward</span>
              <span className="axis-y">Z vertical</span>
              <span className="axis-z">Y lateral</span>
            </div>
          </div>
          <div className="scene-playback-mobile">
            <button className="button button--primary" onClick={playback.playing ? playback.pause : playback.play}>{playback.playing ? 'Pause' : 'Play jump'}</button>
            <button className="button" onClick={playback.restart}>Restart</button>
            <input
              type="range"
              min={0}
              max={duration}
              step={0.01}
              value={playback.time}
              style={{
                '--slider-progress': `${duration > 0 ? (playback.time / duration) * 100 : 0}%`,
              } as React.CSSProperties}
              onChange={(event) => playback.seek(Number(event.target.value))}
              aria-label="Scene timeline"
            />
            <output>{playback.time.toFixed(2)} s</output>
          </div>
        </section>

        <Suspense fallback={<div className="analysis-placeholder">Preparing analysis workspace…</div>}>
          <PlotDashboard
            model={model}
            baselineResult={baselineResult}
            result={result}
            sensitivity={sensitivity}
            currentTime={playback.time}
            onTimeSelect={playback.seek}
          />
        </Suspense>

        <footer className="app-footer">
          <div>
            <strong>{model.modelName}</strong>
            <span>Ridge α={model.training.alpha} · {model.training.acceptedSequences} accepted sequences · no intercept</span>
          </div>
          <div>
            <span>Hill surface: extruded longitudinal profile, not surveyed 3D terrain.</span>
            <span>Displayed distance uses the X/Z flight plane; Y is lateral drift and is reported separately.</span>
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
        <p className="eyebrow">Loading audited model</p>
        <h1>Ski Jump SSM Lab</h1>
      </main>
    )
  }
  return <Dashboard data={data} />
}
