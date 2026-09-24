import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { buildControlSequence, defaultSliders } from '../simulation/controls'
import { calculateMetrics, hillZAt } from '../simulation/metrics'
import type { BaselineData, HillData, ModelData } from '../simulation/model-types'
import { optimizeWindForDistance } from '../simulation/optimization'
import { simulateSSM, simulateToLanding, stateAtTime } from '../simulation/ssm'

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

  it('exposes a -3 to +3 m/s exploratory range for every wind slider', () => {
    model.controls.slice(0, 12).forEach((control) => {
      expect(control.unit).toBe('m/s')
      expect(control.min).toBe(-3)
      expect(control.max).toBe(3)
    })
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

  it('continues the reference flight to its first interpolated hill contact', () => {
    const result = simulateToLanding(model, baseline, hill, defaultSliders(model))
    const endpoint = result.states.at(-1)!
    expect(result.termination).toBe('landing')
    expect(result.extensionSteps).toBe(3)
    expect(result.states.length).toBeGreaterThan(model.trainedHorizon)
    expect(endpoint[2]).toBeCloseTo(hillZAt(hill, endpoint[0]!)!, 9)
    expect(Math.hypot(endpoint[0]!, endpoint[2]!)).toBeCloseTo(210.4166019215, 6)
    result.states.slice(0, -1).forEach((state) => {
      expect(state[2]! - hillZAt(hill, state[0]!)!).toBeGreaterThan(0)
    })
  })

  it('propagates wind controls into the resulting-speed state', () => {
    const defaults = defaultSliders(model)
    const windy = defaults.slice()
    windy[0] = model.controls[0]!.max
    const reference = simulateSSM(model, baseline, defaults)
    const changed = simulateSSM(model, baseline, windy)
    const largestSpeedChange = Math.max(...changed.states.map((state, index) =>
      Math.abs(state[6]! - reference.states[index]![6]!),
    ))
    expect(model.matrices.B[6]!.slice(0, 12).some((value) => Math.abs(value) > 0)).toBe(true)
    expect(largestSpeedChange).toBeGreaterThan(0.01)
  })

  it('finds the exact bounded wind optimum while preserving body settings', () => {
    const sliders = defaultSliders(model)
    sliders[13] = 24.7
    const optimum = optimizeWindForDistance(model, baseline, hill, sliders)

    expect(optimum.sliders.slice(0, 12)).toEqual([
      -3, 3, 3, 3, 3, 3, 3, -3, 3, -3, -3, 3,
    ])
    expect(optimum.sliders.slice(12)).toEqual(sliders.slice(12))

    const simulated = simulateToLanding(model, baseline, hill, optimum.sliders)
    const simulatedEndpoint = simulated.states.at(-1)!
    optimum.endpoint.forEach((value, index) => {
      expect(value).toBeCloseTo(simulatedEndpoint[index]!, 9)
    })
    expect(optimum.displayedDistance).toBeCloseTo(
      Math.hypot(simulatedEndpoint[0]!, simulatedEndpoint[2]!),
      9,
    )
    expect(simulated.termination).toBe('landing')
    expect(simulatedEndpoint[2]).toBeCloseTo(hillZAt(hill, simulatedEndpoint[0]!)!, 9)
    expect(optimum.displayedDistance).toBeGreaterThan(210.4)
  })

  it('includes the certified Planica landing area and full outrun scale', () => {
    expect(hill.landmarks.map((landmark) => landmark.label)).toEqual([
      'K 200 m', 'HS 240 m', 'Outrun U',
    ])
    expect(hill.points.at(-1)![0]).toBeGreaterThan(370)
    expect(hill.points.at(-1)![1]).toBeCloseTo(-145.2, 2)
    expect(hill.points.every((point, index) => index === 0 || point[0] > hill.points[index - 1]![0])).toBe(true)
  })
})
