import type { PlaybackController } from '../hooks/usePlayback'
import { clampSliders, unusualScenario } from '../simulation/controls'
import type { ModelData, Vector } from '../simulation/model-types'
import { ParameterSlider } from './ParameterSlider'

interface ControlPanelProps {
  model: ModelData
  sliders: Vector
  onSlidersChange: (values: Vector) => void
  playback: PlaybackController
  onClose?: () => void
}

const WIND_ZONE_INDICES = {
  takeoff: [0, 3, 6, 9],
  middle: [1, 4, 7, 10],
  landing: [2, 5, 8, 11],
}

function zoneTitle(zone: keyof typeof WIND_ZONE_INDICES): string {
  if (zone === 'takeoff') return 'Takeoff zone'
  if (zone === 'middle') return 'Middle zone'
  return 'Landing zone'
}

export function ControlPanel({
  model,
  sliders,
  onSlidersChange,
  playback,
  onClose,
}: ControlPanelProps) {
  const setSlider = (index: number, value: number) => {
    const next = sliders.slice()
    next[index] = value
    onSlidersChange(next)
  }

  const reset = () =>
    onSlidersChange([
      ...model.controls.map((control) => control.default),
      model.globalAngleOffset.default,
    ])

  const applyPreset = (preset: 'speed' | 'crosswind' | 'angles') => {
    const next = sliders.slice()
    if (preset === 'speed') next[12] = next[12]! + 1
    if (preset === 'crosswind') {
      next[6] = 0.35
      next[7] = -0.35
      next[8] = 0.35
    }
    if (preset === 'angles') next[20] = next[20]! + 0.7
    onSlidersChange(clampSliders(model, next))
  }

  const isUnusual = unusualScenario(model, sliders)

  return (
    <aside className="control-panel" aria-label="Simulation controls">
      <div className="control-panel__title-row">
        <div>
          <p className="eyebrow">Experiment controls</p>
          <h2>Flight parameters</h2>
        </div>
        {onClose && (
          <button className="icon-button mobile-only" onClick={onClose} aria-label="Close controls">
            ×
          </button>
        )}
      </div>

      <section className="control-section playback-controls" aria-labelledby="playback-heading">
        <div className="section-heading" id="playback-heading">
          Playback
          <span>{playback.time.toFixed(2)} s</span>
        </div>
        <div className="playback-buttons">
          <button className="button button--primary" onClick={playback.playing ? playback.pause : playback.play}>
            {playback.playing ? 'Pause' : 'Play'}
          </button>
          <button className="button" onClick={playback.restart}>Restart</button>
          <select
            value={playback.speed}
            onChange={(event) => playback.setSpeed(Number(event.target.value))}
            aria-label="Playback speed"
          >
            {[0.25, 0.5, 1, 1.5, 2].map((value) => (
              <option key={value} value={value}>{value}×</option>
            ))}
          </select>
        </div>
        <input
          className="timeline"
          type="range"
          min={0}
          max={playback.duration}
          step={0.01}
          value={playback.time}
          style={{
            '--slider-progress': `${playback.duration > 0 ? (playback.time / playback.duration) * 100 : 0}%`,
          } as React.CSSProperties}
          onChange={(event) => playback.seek(Number(event.target.value))}
          aria-label="Simulation timeline"
        />
      </section>

      <section className="control-section" aria-labelledby="scenario-heading">
        <div className="section-heading" id="scenario-heading">Scenarios</div>
        <div className="preset-grid">
          <button className="chip" onClick={reset}>Average</button>
          <button className="chip" onClick={() => applyPreset('speed')} title="Add 1 m/s to the current takeoff-speed setting">Speed +1</button>
          <button className="chip" onClick={() => applyPreset('crosswind')}>Crosswind</button>
          <button className="chip" onClick={() => applyPreset('angles')} title="Add 0.7° to the current global angle offset">Angles +0.7°</button>
        </div>
        {isUnusual && (
          <div className="warning-note" role="status">
            Several settings form an unusual combined scenario. Interpret the result cautiously.
          </div>
        )}
      </section>

      <div className="control-scroll">
        {(Object.keys(WIND_ZONE_INDICES) as Array<keyof typeof WIND_ZONE_INDICES>).map((zone) => (
          <details className="control-group" open key={zone}>
            <summary>
              <span>{zoneTitle(zone)}</span>
              <small>wind offsets</small>
            </summary>
            <div className="control-group__body">
              {WIND_ZONE_INDICES[zone].map((index) => (
                <ParameterSlider
                  key={model.controls[index]!.key}
                  definition={model.controls[index]!}
                  value={sliders[index]!}
                  compact
                  onChange={(value) => setSlider(index, value)}
                />
              ))}
            </div>
          </details>
        ))}

        <details className="control-group" open>
          <summary>
            <span>Jumper configuration</span>
            <small>shifted profile means</small>
          </summary>
          <div className="control-group__body">
            {model.controls.slice(12).map((control) => (
              <ParameterSlider
                key={control.key}
                definition={control}
                value={sliders[control.index]!}
                onChange={(value) => setSlider(control.index, value)}
              />
            ))}
            <ParameterSlider
              definition={model.globalAngleOffset}
              value={sliders[20]!}
              onChange={(value) => setSlider(20, value)}
            />
          </div>
        </details>
      </div>
    </aside>
  )
}
