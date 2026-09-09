import { buildControlSequence } from './controls'
import { addVectors, assertFiniteVector, interpolateVector, matrixVectorMultiply } from './math'
import type {
  BaselineData,
  ModelData,
  SimulationResult,
  Vector,
} from './model-types'

export function validateModel(model: ModelData, baseline: BaselineData): void {
  const dimensions: Array<[string, number[][], number, number]> = [
    ['A', model.matrices.A, 14, 14],
    ['B', model.matrices.B, 14, 20],
    ['C', model.matrices.C, 4, 14],
    ['D', model.matrices.D, 4, 20],
  ]
  dimensions.forEach(([name, matrix, rows, columns]) => {
    if (matrix.length !== rows || matrix.some((row) => row.length !== columns)) {
      throw new Error(`${name} must have shape ${rows}x${columns}`)
    }
  })
  if (baseline.states.length !== model.trainedHorizon) {
    throw new Error('Baseline length does not match the trained horizon')
  }
  if (baseline.initialState.length !== 14) {
    throw new Error('Initial state must contain 14 values')
  }
}

export function simulateSSM(
  model: ModelData,
  baseline: BaselineData,
  sliders: Vector,
): SimulationResult {
  validateModel(model, baseline)
  const controls = buildControlSequence(model, baseline, sliders)
  const states: number[][] = [baseline.initialState.slice()]

  for (let timeIndex = 0; timeIndex < model.trainedHorizon - 1; timeIndex += 1) {
    const current = states[timeIndex]!
    const control = controls[timeIndex]!
    const autonomous = matrixVectorMultiply(model.matrices.A, current)
    const forced = matrixVectorMultiply(model.matrices.B, control)
    const next = addVectors(autonomous, forced)
    assertFiniteVector(next, `State ${timeIndex + 1}`)
    states.push(next)
  }

  const observations = states.map((state, timeIndex) => {
    const control = controls[Math.min(timeIndex, controls.length - 1)]!
    return addVectors(
      matrixVectorMultiply(model.matrices.C, state),
      matrixVectorMultiply(model.matrices.D, control),
    )
  })

  const times = states.map((_, index) => index * model.sampleIntervalSeconds)
  return { states, controls, observations, times }
}

export function stateAtTime(result: SimulationResult, timeSeconds: number): Vector {
  const finalTime = result.times[result.times.length - 1] ?? 0
  const clamped = Math.max(0, Math.min(finalTime, timeSeconds))
  if (result.times.length <= 1 || finalTime === 0) return result.states[0]!.slice()
  const step = result.times[1]! - result.times[0]!
  const exactIndex = clamped / step
  const leftIndex = Math.min(Math.floor(exactIndex), result.states.length - 1)
  const rightIndex = Math.min(leftIndex + 1, result.states.length - 1)
  return interpolateVector(
    result.states[leftIndex]!,
    result.states[rightIndex]!,
    exactIndex - leftIndex,
  )
}

