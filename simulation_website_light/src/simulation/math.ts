import type { Matrix, Vector } from './model-types'

export function matrixVectorMultiply(matrix: Matrix, vector: Vector): Vector {
  return matrix.map((row) => {
    if (row.length !== vector.length) {
      throw new Error(
        `Matrix/vector mismatch: row has ${row.length} columns, vector has ${vector.length}`,
      )
    }
    let total = 0
    for (let column = 0; column < row.length; column += 1) {
      total += row[column]! * vector[column]!
    }
    return total
  })
}

export function addVectors(left: Vector, right: Vector): Vector {
  if (left.length !== right.length) {
    throw new Error(`Vector mismatch: ${left.length} !== ${right.length}`)
  }
  return left.map((value, index) => value + right[index]!)
}

export function assertFiniteVector(vector: Vector, label: string): void {
  const invalidIndex = vector.findIndex((value) => !Number.isFinite(value))
  if (invalidIndex >= 0) {
    throw new Error(`${label} contains a non-finite value at index ${invalidIndex}`)
  }
}

export function interpolateVector(left: Vector, right: Vector, fraction: number): Vector {
  const clamped = Math.max(0, Math.min(1, fraction))
  return left.map((value, index) => value + (right[index]! - value) * clamped)
}

