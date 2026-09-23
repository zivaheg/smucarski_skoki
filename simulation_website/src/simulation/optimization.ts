import { defaultSliders } from './controls'
import type { BaselineData, HillData, ModelData, Vector } from './model-types'
import { interpolateHillContact, simulateSSM, simulateToLanding } from './ssm'

export const WIND_CONTROL_COUNT = 12

export interface WindOptimizationResult {
  sliders: Vector
  endpoint: Vector
  displayedDistance: number
}

/**
 * Find the longest hill-contact distance among all bounded wind corners.
 *
 * The state at every fixed timestep is affine in the 12 wind offsets. We run
 * one reference trajectory plus one unit-response trajectory per wind input,
 * then reconstruct all 4,096 corners cheaply. Landing makes the objective
 * piecewise nonlinear, so every candidate is checked for its first hill
 * intersection rather than maximizing a fixed-horizon endpoint.
 */
export function optimizeWindForDistance(
  model: ModelData,
  baseline: BaselineData,
  hill: HillData,
  currentSliders: Vector = defaultSliders(model),
): WindOptimizationResult {
  const reference = currentSliders.slice()
  for (let index = 0; index < WIND_CONTROL_COUNT; index += 1) {
    reference[index] = 0
  }

  const extensionSteps = (model.training.maximumSequenceLength ?? 163) - model.trainedHorizon
  const referenceResult = simulateSSM(model, baseline, reference, extensionSteps)
  const unitResults = Array.from({ length: WIND_CONTROL_COUNT }, (_, index) => {
    const perturbed = reference.slice()
    perturbed[index] = 1
    return simulateSSM(model, baseline, perturbed, extensionSteps)
  })

  const positionAt = (timeIndex: number, candidate: Vector): Vector =>
    [0, 1, 2].map((stateIndex) =>
      referenceResult.states[timeIndex]![stateIndex]!
      + unitResults.reduce(
        (sum, unitResult, windIndex) =>
          sum
          + (
            unitResult.states[timeIndex]![stateIndex]!
            - referenceResult.states[timeIndex]![stateIndex]!
          ) * candidate[windIndex]!,
        0,
      ),
    )

  let bestSliders: Vector | null = null
  let bestDistance = Number.NEGATIVE_INFINITY
  const cornerCount = 2 ** WIND_CONTROL_COUNT

  for (let corner = 0; corner < cornerCount; corner += 1) {
    const candidate = reference.slice()
    for (let index = 0; index < WIND_CONTROL_COUNT; index += 1) {
      const definition = model.controls[index]!
      candidate[index] = (corner & (1 << index)) === 0
        ? definition.min
        : definition.max
    }

    let previous = positionAt(0, candidate)
    for (let timeIndex = 1; timeIndex < referenceResult.states.length; timeIndex += 1) {
      const current = positionAt(timeIndex, candidate)
      const contact = interpolateHillContact(hill, previous, current)
      if (contact) {
        const distance = Math.hypot(contact.state[0]!, contact.state[2]!)
        if (distance > bestDistance) {
          bestDistance = distance
          bestSliders = candidate
        }
        break
      }
      previous = current
    }
  }

  if (bestSliders === null) {
    throw new Error('No bounded wind scenario reached the hill within the supported flight duration')
  }

  const bestResult = simulateToLanding(model, baseline, hill, bestSliders)
  const endpoint = bestResult.states.at(-1)!.slice()
  return {
    sliders: bestSliders,
    endpoint,
    displayedDistance: Math.hypot(endpoint[0]!, endpoint[2]!),
  }
}
