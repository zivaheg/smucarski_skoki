import type {
  HillData,
  JumpMetrics,
  SimulationResult,
  StateSequence,
} from './model-types'

export function hillZAt(hill: HillData, x: number): number | null {
  const points = hill.points
  if (points.length === 0) return null
  if (x <= points[0]![0]) return points[0]![1]
  if (x >= points[points.length - 1]![0]) return points[points.length - 1]![1]
  for (let index = 1; index < points.length; index += 1) {
    const left = points[index - 1]!
    const right = points[index]!
    if (x <= right[0]) {
      const span = right[0] - left[0]
      const fraction = span === 0 ? 0 : (x - left[0]) / span
      return left[1] + (right[1] - left[1]) * fraction
    }
  }
  return points[points.length - 1]![1]
}

export function displayedDistance(states: StateSequence): number {
  const endpoint = states[states.length - 1]!
  return Math.hypot(endpoint[0]!, endpoint[2]!)
}

export function calculateMetrics(
  result: SimulationResult,
  hill: HillData,
  baselineDistance: number,
): JumpMetrics {
  const endpointState = result.states[result.states.length - 1]!
  const endpoint: [number, number, number] = [
    endpointState[0]!,
    endpointState[1]!,
    endpointState[2]!,
  ]
  const distance = Math.hypot(endpoint[0], endpoint[2])
  const clearances = result.states
    .map((state) => {
      const ground = hillZAt(hill, state[0]!)
      return ground === null ? null : state[2]! - ground
    })
    .filter((value): value is number => value !== null)
  const landingGround = hillZAt(hill, endpoint[0])
  return {
    endpoint,
    displayedDistance: distance,
    distanceDelta: distance - baselineDistance,
    maxZ: Math.max(...result.states.map((state) => state[2]!)),
    maxAbsY: Math.max(...result.states.map((state) => Math.abs(state[1]!))),
    minimumClearance: clearances.length > 0 ? Math.min(...clearances) : null,
    landingClearance: landingGround === null ? null : endpoint[2] - landingGround,
    finite: result.states.every((state) => state.every(Number.isFinite)),
  }
}
