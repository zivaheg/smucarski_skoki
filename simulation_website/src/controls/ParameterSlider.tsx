import type { ControlDefinition, GlobalAngleOffsetDefinition } from '../simulation/model-types'

interface ParameterSliderProps {
  definition: ControlDefinition | GlobalAngleOffsetDefinition
  value: number
  onChange: (value: number) => void
  compact?: boolean
}

function decimals(step: number): number {
  if (step >= 1) return 0
  return Math.max(1, Math.ceil(-Math.log10(step)))
}

export function ParameterSlider({
  definition,
  value,
  onChange,
  compact = false,
}: ParameterSliderProps) {
  const display = value.toFixed(decimals(definition.step))
  const percent = ((value - definition.min) / (definition.max - definition.min)) * 100
  const kind = 'valueKind' in definition ? definition.valueKind : 'offset'
  return (
    <label className={`parameter-slider${compact ? ' parameter-slider--compact' : ''}`}>
      <span className="parameter-slider__header">
        <span>{definition.label}</span>
        <output>
          {display} <small>{definition.unit}</small>
        </output>
      </span>
      <input
        type="range"
        min={definition.min}
        max={definition.max}
        step={definition.step}
        value={value}
        aria-label={`${definition.label}, ${kind}`}
        style={{ '--slider-progress': `${percent}%` } as React.CSSProperties}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      {!compact && (
        <span className="parameter-slider__limits">
          <span>{definition.min}</span>
          <span>{kind === 'mean' ? 'profile mean' : 'baseline offset'}</span>
          <span>{definition.max}</span>
        </span>
      )}
    </label>
  )
}

