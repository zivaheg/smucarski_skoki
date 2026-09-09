import type {
  BaselineData,
  ControlSequence,
  ModelData,
  Vector,
} from './model-types'
import { assertFiniteVector } from './math'

export const MODEL_CONTROL_COUNT = 20
export const UI_SLIDER_COUNT = 21

export function buildControlSequence(
  model: ModelData,
  baseline: BaselineData,
  sliders: Vector,
): ControlSequence {
  if (sliders.length !== UI_SLIDER_COUNT) {
    throw new Error(`Expected ${UI_SLIDER_COUNT} slider values, received ${sliders.length}`)
  }
  if (model.baselineWindMeans.length !== 12) {
    throw new Error('Expected 12 baseline wind means')
  }
  if (baseline.bodyMeans.length !== 8) {
    throw new Error('Expected eight baseline body means')
  }

  return baseline.bodyProfiles.map((profile, timeIndex) => {
    if (profile.length !== 8) {
      throw new Error(`Body profile ${timeIndex} does not have eight values`)
    }
    const controls = new Array<number>(MODEL_CONTROL_COUNT)
    for (let index = 0; index < 12; index += 1) {
      controls[index] = model.baselineWindMeans[index]! + sliders[index]!
    }
    for (let bodyIndex = 0; bodyIndex < 8; bodyIndex += 1) {
      const controlIndex = bodyIndex + 12
      const selectedMean = sliders[controlIndex]!
      const profileOffset = selectedMean - baseline.bodyMeans[bodyIndex]!
      const globalAngleOffset = bodyIndex === 0 ? 0 : sliders[20]!
      controls[controlIndex] = profile[bodyIndex]! + profileOffset + globalAngleOffset
    }
    assertFiniteVector(controls, `Control row ${timeIndex}`)
    return controls
  })
}

export function clampSliders(model: ModelData, values: Vector): Vector {
  if (values.length !== UI_SLIDER_COUNT) return defaultSliders(model)
  const clamped = values.slice()
  model.controls.forEach((control) => {
    clamped[control.index] = Math.max(
      control.min,
      Math.min(control.max, values[control.index]!),
    )
  })
  const global = model.globalAngleOffset
  clamped[20] = Math.max(global.min, Math.min(global.max, values[20]!))
  return clamped
}

export function defaultSliders(model: ModelData): Vector {
  return [
    ...model.controls.map((control) => control.default),
    model.globalAngleOffset.default,
  ]
}

export function scenarioExtremeness(model: ModelData, sliders: Vector): number {
  const definitions = [...model.controls, model.globalAngleOffset]
  let sumSquares = 0
  definitions.forEach((definition, index) => {
    const halfRange = Math.max((definition.max - definition.min) / 2, Number.EPSILON)
    const centre = (definition.max + definition.min) / 2
    const normalized = (sliders[index]! - centre) / halfRange
    sumSquares += normalized * normalized
  })
  return Math.sqrt(sumSquares / definitions.length)
}

export function unusualScenario(model: ModelData, sliders: Vector): boolean {
  const definitions = [...model.controls, model.globalAngleOffset]
  const nearLimits = definitions.filter((definition, index) => {
    const range = definition.max - definition.min
    const distanceToEdge = Math.min(
      sliders[index]! - definition.min,
      definition.max - sliders[index]!,
    )
    return distanceToEdge / range < 0.08
  }).length
  return nearLimits >= 3 || scenarioExtremeness(model, sliders) > 0.78
}

