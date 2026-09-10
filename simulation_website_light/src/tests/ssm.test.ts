import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { buildControlSequence, defaultSliders } from '../simulation/controls'
import { calculateMetrics } from '../simulation/metrics'
import type { BaselineData, HillData, ModelData } from '../simulation/model-types'
import { simulateSSM, stateAtTime } from '../simulation/ssm'

function asset<T>(name: string): T {
  const file = resolve(process.cwd(), 'public', 'data', name)
  return JSON.parse(readFileSync(file, 'utf-8')) as T
}

const model = asset<ModelData>('model.json')
const baseline = asset<BaselineData>('baseline-flight.json')
const hill = asset<HillData>('hill-profile.json')

describe('production state-space model', () => {
  it('preserves the audited matrix dimensions and ordering', () => {
    expect(model.matrices.A).toHaveLength(14)
    expect(model.matrices.A.every((row) => row.length === 14)).toBe(true)
    expect(model.matrices.B.every((row) => row.length === 20)).toBe(true)
    expect(model.matrices.C).toHaveLength(4)
    expect(model.matrices.D.every((row) => row.length === 20)).toBe(true)
    expect(model.controls.map((control) => control.key)).toEqual([
      'windSpeedTakeoff',
      'windSpeedMiddle',
      'windSpeedLanding',
      'windTangentTakeoff',
      'windTangentMiddle',
      'windTangentLanding',
      'crosswindTakeoff',
      'crosswindMiddle',
      'crosswindLanding',
      'turbulenceTakeoff',
      'turbulenceMiddle',
      'turbulenceLanding',
      'speed',
      'opening',
      'rollLeft',
      'rollRight',
      'yawLeft',
      'yawRight',
      'stallLeft',
      'stallRight',
    ])
  })

  it('matches the Python default trajectory reference', () => {
    const result = simulateSSM(model, baseline, defaultSliders(model))
    const endpoint = result.states.at(-1)!.slice(0, 3)
    endpoint.forEach((value, index) => {
      expect(value).toBeCloseTo(baseline.referenceSimulation.endpoint[index]!, 9)
    })
    const metrics = calculateMetrics(result, hill, baseline.referenceSimulation.displayedDistance)
    expect(metrics.displayedDistance).toBeCloseTo(
      baseline.referenceSimulation.displayedDistance,
      9,
    )
    expect(result.states).toHaveLength(139)
    expect(result.observations.every((row) => row.length === 4)).toBe(true)
  })

  it('constructs the first control row exactly like Python', () => {
    const controls = buildControlSequence(model, baseline, defaultSliders(model))
    controls[0]!.forEach((value, index) => {
      expect(value).toBeCloseTo(baseline.referenceSimulation.firstControl[index]!, 12)
    })
  })

  it('keeps every documented single-control extreme finite', () => {
    const defaults = defaultSliders(model)
    const definitions = [...model.controls, model.globalAngleOffset]
    definitions.forEach((definition, index) => {
      for (const value of [definition.min, definition.max]) {
        const sliders = defaults.slice()
        sliders[index] = value
        const result = simulateSSM(model, baseline, sliders)
        expect(result.states.every((state) => state.every(Number.isFinite))).toBe(true)
      }
    })
  })

  it('interpolates state playback on the physical 0.05 second timeline', () => {
    const result = simulateSSM(model, baseline, defaultSliders(model))
    expect(result.times[1]).toBeCloseTo(0.05, 12)
    const midpoint = stateAtTime(result, 0.025)
    midpoint.forEach((value, index) => {
      expect(value).toBeCloseTo((result.states[0]![index]! + result.states[1]![index]!) / 2, 12)
    })
  })
})

