import type { EChartsOption } from 'echarts'
import { useMemo, useState } from 'react'
import type { SensitivityResult } from '../simulation/sensitivity'
import type { ModelData, SimulationResult } from '../simulation/model-types'
import { EChartPanel } from './EChartPanel'

export type PlotId =
  | 'side'
  | 'top'
  | 'position'
  | 'velocity'
  | 'angles'
  | 'wind'
  | 'observations'
  | 'sensitivity'
  | 'matrixA'
  | 'matrixB'
  | 'matrixC'
  | 'matrixD'

interface PlotDefinition {
  id: PlotId
  label: string
  group: string
}

const PLOTS: PlotDefinition[] = [
  { id: 'side', label: 'Side trajectory · X/Z', group: 'Trajectory' },
  { id: 'top', label: 'Top trajectory · X/Y', group: 'Trajectory' },
  { id: 'position', label: 'Position over time', group: 'Motion' },
  { id: 'velocity', label: 'Velocity over time', group: 'Motion' },
  { id: 'angles', label: 'Body angles', group: 'Configuration' },
  { id: 'wind', label: 'Wind controls', group: 'Environment' },
  { id: 'observations', label: 'SSM observations', group: 'Model' },
  { id: 'sensitivity', label: 'Distance sensitivity', group: 'Outcome' },
  { id: 'matrixA', label: 'A · state dynamics', group: 'Matrices' },
  { id: 'matrixB', label: 'B · control influence', group: 'Matrices' },
  { id: 'matrixC', label: 'C · observations', group: 'Matrices' },
  { id: 'matrixD', label: 'D · feedthrough', group: 'Matrices' },
]

const COLORS = ['#44d7ef', '#ff9b55', '#a78bfa', '#6ee7b7', '#f5c75f', '#f472b6', '#7dd3fc']

interface PlotDashboardProps {
  model: ModelData
  baselineResult: SimulationResult
  result: SimulationResult
  sensitivity: SensitivityResult[]
  currentTime: number
  onTimeSelect: (time: number) => void
}

function axisStyle(name: string) {
  return {
    name,
    nameTextStyle: { color: '#90a9b4' },
    axisLine: { lineStyle: { color: '#35505d' } },
    axisLabel: { color: '#9fb3bd' },
    splitLine: { lineStyle: { color: 'rgba(112, 146, 160, 0.14)' } },
  }
}

function lineOption(
  xName: string,
  yName: string,
  series: Array<{ name: string; data: number[][]; color?: string; dashed?: boolean }>,
  currentTime?: number,
): EChartsOption {
  return {
    animation: false,
    backgroundColor: 'transparent',
    color: COLORS,
    grid: { left: 54, right: 20, top: 42, bottom: 44 },
    legend: { top: 4, textStyle: { color: '#c6d6dc' }, type: 'scroll' },
    tooltip: { trigger: 'axis', backgroundColor: '#10232d', borderColor: '#335260', textStyle: { color: '#eef7fa' } },
    xAxis: { type: 'value', ...axisStyle(xName) },
    yAxis: { type: 'value', ...axisStyle(yName), scale: true },
    dataZoom: [{ type: 'inside' }],
    series: series.map((item, index) => ({
      name: item.name,
      type: 'line',
      data: item.data,
      symbol: 'none',
      smooth: 0.16,
      lineStyle: {
        width: item.dashed ? 1.5 : 2,
        type: item.dashed ? 'dashed' : 'solid',
        color: item.color ?? COLORS[index % COLORS.length],
        opacity: item.dashed ? 0.7 : 1,
      },
      markLine:
        currentTime === undefined || index !== 0
          ? undefined
          : {
              silent: true,
              symbol: 'none',
              label: { show: false },
              lineStyle: { color: '#ffffff', opacity: 0.36, width: 1 },
              data: [{ xAxis: currentTime }],
            },
    })),
  }
}

function matrixOption(
  matrix: number[][],
  rowLabels: string[],
  columnLabels: string[],
): EChartsOption {
  const values = matrix.flatMap((row, rowIndex) =>
    row.map((value, columnIndex) => [columnIndex, rowIndex, value]),
  )
  const extent = Math.max(...matrix.flat().map(Math.abs), Number.EPSILON)
  return {
    animation: false,
    backgroundColor: 'transparent',
    grid: { left: 112, right: 72, top: 24, bottom: 96 },
    tooltip: {
      formatter: (params: any) => {
        const [column, row, value] = params.value as [number, number, number]
        return `${rowLabels[row]} ← ${columnLabels[column]}<br/><strong>${value.toExponential(4)}</strong>`
      },
      backgroundColor: '#10232d',
      borderColor: '#335260',
      textStyle: { color: '#eef7fa' },
    },
    xAxis: { type: 'category', data: columnLabels, axisLabel: { color: '#9fb3bd', rotate: 45, fontSize: 10 }, axisLine: { lineStyle: { color: '#35505d' } } },
    yAxis: { type: 'category', data: rowLabels, axisLabel: { color: '#9fb3bd', fontSize: 10 }, axisLine: { lineStyle: { color: '#35505d' } } },
    visualMap: {
      min: -extent,
      max: extent,
      calculable: true,
      orient: 'vertical',
      right: 4,
      top: 'middle',
      textStyle: { color: '#9fb3bd' },
      inRange: { color: ['#2563a8', '#0b1720', '#d46b35'] },
    },
    series: [{ type: 'heatmap', data: values, emphasis: { itemStyle: { borderColor: '#fff', borderWidth: 1 } } }],
  }
}

function createOption(
  id: PlotId,
  model: ModelData,
  baseline: SimulationResult,
  result: SimulationResult,
  sensitivity: SensitivityResult[],
  currentTime: number,
): EChartsOption {
  const timed = (sequence: number[][], index: number) =>
    result.times.map((time, row) => [time, sequence[row]![index]!])
  const comparison = (xIndex: number, yIndex: number) => [
    { name: 'Modified', data: result.states.map((state) => [state[xIndex]!, state[yIndex]!]), color: '#ff9b55' },
    { name: 'Baseline', data: baseline.states.map((state) => [state[xIndex]!, state[yIndex]!]), color: '#44d7ef', dashed: true },
  ]

  if (id === 'side') return lineOption('X [m]', 'Z [m]', comparison(0, 2))
  if (id === 'top') return lineOption('X [m]', 'Y [m]', comparison(0, 1))
  if (id === 'position') {
    return lineOption('Time [s]', 'Position [m]', [
      { name: 'X', data: timed(result.states, 0) },
      { name: 'Y', data: timed(result.states, 1) },
      { name: 'Z', data: timed(result.states, 2) },
      { name: 'Baseline X', data: timed(baseline.states, 0), dashed: true },
    ], currentTime)
  }
  if (id === 'velocity') {
    return lineOption('Time [s]', 'Velocity [m/s]', [3, 4, 5].map((index, color) => ({
      name: model.states[index]!.label,
      data: timed(result.states, index),
      color: COLORS[color],
    })), currentTime)
  }
  if (id === 'angles') {
    return lineOption('Time [s]', 'Angle [deg]', [7, 8, 9, 10, 11, 12, 13].map((index) => ({
      name: model.states[index]!.label,
      data: timed(result.states, index),
    })), currentTime)
  }
  if (id === 'wind') {
    return lineOption('Time [s]', 'Wind control', model.controls.slice(0, 12).map((control) => ({
      name: `${control.group} · ${control.label}`,
      data: timed(result.controls, control.index),
    })), currentTime)
  }
  if (id === 'observations') {
    return lineOption('Time [s]', 'Observation', model.observations.map((observation) => ({
      name: observation.label,
      data: timed(result.observations, observation.index),
    })), currentTime)
  }
  if (id === 'sensitivity') {
    const sorted = [...sensitivity].sort((left, right) => left.distanceDelta - right.distanceDelta)
    return {
      animationDuration: 250,
      backgroundColor: 'transparent',
      grid: { left: 132, right: 28, top: 20, bottom: 46 },
      tooltip: {
        formatter: (params: any) => `${params.name}<br/><strong>${Number(params.value).toFixed(3)} m</strong> per slider step`,
        backgroundColor: '#10232d', borderColor: '#335260', textStyle: { color: '#eef7fa' },
      },
      xAxis: { type: 'value', ...axisStyle('Δ displayed distance [m]') },
      yAxis: { type: 'category', data: sorted.map((item) => item.label), axisLabel: { color: '#9fb3bd', fontSize: 10 }, axisLine: { lineStyle: { color: '#35505d' } } },
      series: [{ type: 'bar', data: sorted.map((item) => ({ value: item.distanceDelta, itemStyle: { color: item.distanceDelta >= 0 ? '#ff9b55' : '#44d7ef' } })) }],
    }
  }

  const matrices = {
    matrixA: [model.matrices.A, model.states, model.states],
    matrixB: [model.matrices.B, model.states, model.controls],
    matrixC: [model.matrices.C, model.observations, model.states],
    matrixD: [model.matrices.D, model.observations, model.controls],
  } as const
  const [matrix, rows, columns] = matrices[id as keyof typeof matrices]
  return matrixOption(matrix, rows.map((item) => item.label), columns.map((item) => item.label))
}

export function PlotDashboard({
  model,
  baselineResult,
  result,
  sensitivity,
  currentTime,
  onTimeSelect,
}: PlotDashboardProps) {
  const [selected, setSelected] = useState<PlotId[]>(['side', 'position', 'sensitivity', 'matrixB'])

  const options = useMemo(
    () =>
      selected.map((id) => ({
        id,
        option: createOption(id, model, baselineResult, result, sensitivity, currentTime),
      })),
    [baselineResult, currentTime, model, result, selected, sensitivity],
  )

  const toggle = (id: PlotId) => {
    setSelected((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id],
    )
  }

  return (
    <section className="analysis-section" aria-labelledby="analysis-title">
      <div className="analysis-header">
        <div>
          <p className="eyebrow">Linked analysis</p>
          <h2 id="analysis-title">Plots & matrices</h2>
        </div>
        <details className="plot-picker">
          <summary>{selected.length} plots visible</summary>
          <div className="plot-picker__menu">
            {PLOTS.map((plot) => (
              <label key={plot.id}>
                <input
                  type="checkbox"
                  checked={selected.includes(plot.id)}
                  onChange={() => toggle(plot.id)}
                />
                <span>{plot.label}</span>
                <small>{plot.group}</small>
              </label>
            ))}
          </div>
        </details>
      </div>
      {selected.length === 0 ? (
        <div className="empty-state">Choose one or more plots from the menu.</div>
      ) : (
        <div className="plot-grid">
          {options.map(({ id, option }) => {
            const definition = PLOTS.find((plot) => plot.id === id)!
            return (
              <article className="plot-card" key={id}>
                <div className="plot-card__header">
                  <div>
                    <strong>{definition.label}</strong>
                    <small>{definition.group}</small>
                  </div>
                  <button className="icon-button" onClick={() => toggle(id)} aria-label={`Hide ${definition.label}`}>×</button>
                </div>
                <EChartPanel option={option} onTimeSelect={onTimeSelect} />
              </article>
            )
          })}
        </div>
      )}
    </section>
  )
}

