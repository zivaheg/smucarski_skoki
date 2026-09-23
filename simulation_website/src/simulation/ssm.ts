import { buildControlSequence } from './controls'
import { hillZAt } from './metrics'
import { addVectors, assertFiniteVector, interpolateVector, matrixVectorMultiply } from './math'
import type {
  BaselineData,
  HillData,
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
  if ((model.training.maximumSequenceLength ?? 163) < model.trainedHorizon) {
    throw new Error('Maximum sequence length cannot be shorter than the reference horizon')
  }
}

function createResult(
  model: ModelData,
  states: number[][],
  controls: number[][],
  times: number[],
  termination: SimulationResult['termination'],
  extensionSteps: number,
): SimulationResult {
  const observations = states.map((state, timeIndex) => {
    const control = controls[Math.min(timeIndex, controls.length - 1)]!
    return addVectors(
      matrixVectorMultiply(model.matrices.C, state),
      matrixVectorMultiply(model.matrices.D, control),
    )
  })
  return { states, controls, observations, times, termination, extensionSteps }
}

export function simulateSSM(
  model: ModelData,
  baseline: BaselineData,
  sliders: Vector,
  additionalSteps = 0,
): SimulationResult {
  validateModel(model, baseline)
  if (!Number.isInteger(additionalSteps) || additionalSteps < 0) {
    throw new Error('Additional simulation steps must be a non-negative integer')
  }
  const baselineControls = buildControlSequence(model, baseline, sliders)
  const stateCount = model.trainedHorizon + additionalSteps
  const controls = Array.from({ length: stateCount }, (_, index) =>
    baselineControls[Math.min(index, baselineControls.length - 1)]!.slice(),
  )
  const states: number[][] = [baseline.initialState.slice()]

  for (let timeIndex = 0; timeIndex < stateCount - 1; timeIndex += 1) {
    const current = states[timeIndex]!
    const control = controls[timeIndex]!
    const autonomous = matrixVectorMultiply(model.matrices.A, current)
    const forced = matrixVectorMultiply(model.matrices.B, control)
    const next = addVectors(autonomous, forced)
    assertFiniteVector(next, `State ${timeIndex + 1}`)
    states.push(next)
  }

  const times = states.map((_, index) => index * model.sampleIntervalSeconds)
  return createResult(
    model,
    states,
    controls,
    times,
    additionalSteps === 0 ? 'horizon' : 'extension-limit',
    additionalSteps,
  )
}

function clearance(hill: HillData, state: Vector): number | null {
  const ground = hillZAt(hill, state[0]!)
  return ground === null ? null : state[2]! - ground
}

export function interpolateHillContact(
  hill: HillData,
  airborne: Vector,
  grounded: Vector,
): { state: Vector; fraction: number } | null {
  const airborneClearance = clearance(hill, airborne)
  const groundedClearance = clearance(hill, grounded)
  if (
    airborneClearance === null
    || groundedClearance === null
    || airborneClearance <= 0
    || groundedClearance > 0
  ) return null

  let low = 0
  let high = 1
  for (let iteration = 0; iteration < 36; iteration += 1) {
    const middle = (low + high) / 2
    const middleState = interpolateVector(airborne, grounded, middle)
    const middleClearance = clearance(hill, middleState)
    if (middleClearance !== null && middleClearance > 0) low = middle
    else high = middle
  }
  const fraction = (low + high) / 2
  const state = interpolateVector(airborne, grounded, fraction)
  const ground = hillZAt(hill, state[0]!)
  if (ground !== null) state[2] = ground
  return { state, fraction }
}

export function simulateToLanding(
  model: ModelData,
  baseline: BaselineData,
  hill: HillData,
  sliders: Vector,
): SimulationResult {
  const maximumExtension = (model.training.maximumSequenceLength ?? 163) - model.trainedHorizon
  const extended = simulateSSM(model, baseline, sliders, maximumExtension)

  for (let index = 1; index < extended.states.length; index += 1) {
    const contact = interpolateHillContact(
      hill,
      extended.states[index - 1]!,
      extended.states[index]!,
    )
    if (!contact) continue

    const states = extended.states.slice(0, index)
    states.push(contact.state)
    const controls = extended.controls.slice(0, states.length)
    const times = extended.times.slice(0, index)
    times.push(
      extended.times[index - 1]!
      + contact.fraction * model.sampleIntervalSeconds,
    )
    return createResult(
      model,
      states,
      controls,
      times,
      'landing',
      Math.max(0, index - (model.trainedHorizon - 1)),
    )
  }

  return extended
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
