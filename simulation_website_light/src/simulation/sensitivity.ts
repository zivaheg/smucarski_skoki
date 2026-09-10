import type { BaselineData, HillData, ModelData, Vector } from './model-types'
import { calculateMetrics } from './metrics'
import { simulateSSM } from './ssm'

export interface SensitivityResult {
  index: number
  label: string
  unit: string
  step: number
  distanceDelta: number
}

export function calculateSensitivity(
  model: ModelData,
  baseline: BaselineData,
  hill: HillData,
  sliders: Vector,
): SensitivityResult[] {
  const baseResult = simulateSSM(model, baseline, sliders)
  const baseDistance = calculateMetrics(baseResult, hill, 0).displayedDistance
  const definitions = [...model.controls, model.globalAngleOffset]
  return definitions.map((definition, index) => {
    const adjusted = sliders.slice()
    const positive = Math.min(definition.max, sliders[index]! + definition.step)
    const usableStep = positive - sliders[index]!
    if (usableStep === 0) {
      adjusted[index] = Math.max(definition.min, sliders[index]! - definition.step)
    } else {
      adjusted[index] = positive
    }
    const result = simulateSSM(model, baseline, adjusted)
    const distance = calculateMetrics(result, hill, baseDistance).displayedDistance
    return {
      index,
      label: definition.label,
      unit: definition.unit,
      step: adjusted[index]! - sliders[index]!,
      distanceDelta: distance - baseDistance,
    }
  })
}

