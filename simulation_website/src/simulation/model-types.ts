export type Matrix = number[][]
export type Vector = number[]
export type StateSequence = number[][]
export type ControlSequence = number[][]

export interface NamedDimension {
  index: number
  key: string
  label: string
  unit: string
}

export interface ControlDefinition extends NamedDimension {
  group: 'takeoff' | 'middle' | 'landing' | 'body'
  valueKind: 'offset' | 'mean'
  min: number
  max: number
  step: number
  default: number
  baseline: number
}

export interface GlobalAngleOffsetDefinition {
  index: 20
  key: string
  label: string
  unit: string
  min: number
  max: number
  step: number
  default: number
  appliesToControlIndices: number[]
}

export interface ModelData {
  schemaVersion: number
  modelName: string
  description: string
  sampleIntervalSeconds: number
  trainedHorizon: number
  training: {
    acceptedSequences: number
    excludedSequenceIndex: number
    regression: string
    alpha: number
    fitIntercept: boolean
    reportedCrossValidationErrorMetres: number
    reportedAverageFlightBaselineErrorMetres: number
  }
  states: NamedDimension[]
  controls: ControlDefinition[]
  observations: NamedDimension[]
  globalAngleOffset: GlobalAngleOffsetDefinition
  baselineWindMeans: number[]
  matrices: {
    A: Matrix
    B: Matrix
    C: Matrix
    D: Matrix
  }
  diagnostics: {
    spectralRadiusA: number
    rankA: number
    rankB: number
    rankC: number
  }
  warnings: string[]
  provenance: Record<string, { source: string; sha256: string }>
}

export interface BaselineData {
  schemaVersion: number
  sampleIntervalSeconds: number
  initialState: Vector
  states: StateSequence
  bodyProfiles: number[][]
  bodyMeans: Vector
  defaultBodySliderValues: Vector
  defaultSliderValues: Vector
  referenceSimulation: {
    endpoint: Vector
    displayedDistance: number
    maxZ: number
    maxAbsY: number
    finalState: Vector
    firstControl: Vector
  }
}

export interface HillData {
  schemaVersion: number
  coordinateSystem: {
    x: string
    y: string
    z: string
  }
  surfaceHalfWidth: number
  distanceMarkers: number[]
  points: Array<[number, number]>
  note: string
}

export interface SimulationResult {
  states: StateSequence
  controls: ControlSequence
  observations: StateSequence
  times: number[]
}

export interface JumpMetrics {
  endpoint: [number, number, number]
  displayedDistance: number
  distanceDelta: number
  maxZ: number
  maxAbsY: number
  minimumClearance: number | null
  landingClearance: number | null
  finite: boolean
}

export interface AppData {
  model: ModelData
  baseline: BaselineData
  hill: HillData
}

