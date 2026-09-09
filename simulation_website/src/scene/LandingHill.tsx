import { Html, Line } from '@react-three/drei'
import { useEffect, useMemo } from 'react'
import { BufferGeometry, Float32BufferAttribute } from 'three'
import type { HillData } from '../simulation/model-types'
import { hillZAt } from '../simulation/metrics'

interface LandingHillProps {
  hill: HillData
  showLabels: boolean
}

export function LandingHill({ hill, showLabels }: LandingHillProps) {
  const geometry = useMemo(() => {
    const vertices: number[] = []
    const indices: number[] = []
    hill.points.forEach(([x, z]) => {
      vertices.push(x, z, -hill.surfaceHalfWidth)
      vertices.push(x, z, hill.surfaceHalfWidth)
    })
    for (let index = 0; index < hill.points.length - 1; index += 1) {
      const a = index * 2
      const b = a + 1
      const c = a + 2
      const d = a + 3
      indices.push(a, b, c, b, d, c)
    }
    const mesh = new BufferGeometry()
    mesh.setAttribute('position', new Float32BufferAttribute(vertices, 3))
    mesh.setIndex(indices)
    mesh.computeVertexNormals()
    return mesh
  }, [hill])

  useEffect(() => () => geometry.dispose(), [geometry])

  const leftEdge = hill.points.map(([x, z]) => [x, z + 0.08, -hill.surfaceHalfWidth] as const)
  const centre = hill.points.map(([x, z]) => [x, z + 0.1, 0] as const)
  const rightEdge = hill.points.map(([x, z]) => [x, z + 0.08, hill.surfaceHalfWidth] as const)
  const laneOffsets = [-8, -4, 4, 8].filter((offset) => Math.abs(offset) < hill.surfaceHalfWidth - 1)
  const rampStart: [number, number] = [-32, 19]
  const rampEnd: [number, number] = [-1.5, 1.15]
  const rampLength = Math.hypot(rampEnd[0] - rampStart[0], rampEnd[1] - rampStart[1])
  const rampAngle = Math.atan2(rampEnd[1] - rampStart[1], rampEnd[0] - rampStart[0])
  const rampCentre: [number, number] = [
    (rampStart[0] + rampEnd[0]) / 2,
    (rampStart[1] + rampEnd[1]) / 2,
  ]

  return (
    <group>
      <mesh geometry={geometry} receiveShadow>
        <meshStandardMaterial color="#dcecf0" roughness={0.96} metalness={0.01} />
      </mesh>
      <Line points={leftEdge} color="#477a88" lineWidth={1.2} transparent opacity={0.72} />
      <Line points={centre} color="#4c7e8b" lineWidth={1.2} transparent opacity={0.58} />
      <Line points={rightEdge} color="#477a88" lineWidth={1.2} transparent opacity={0.72} />
      {laneOffsets.map((offset) => (
        <Line
          key={offset}
          points={hill.points.map(([x, z]) => [x, z + 0.1, offset] as const)}
          color="#7ea2aa"
          lineWidth={0.7}
          transparent
          opacity={0.38}
        />
      ))}

      {hill.distanceMarkers.map((x) => {
        const z = hillZAt(hill, x) ?? 0
        return (
          <group key={x}>
            <Line
              points={[
                [x, z + 0.12, -hill.surfaceHalfWidth],
                [x, z + 0.12, hill.surfaceHalfWidth],
              ]}
              color="#547b85"
              lineWidth={0.8}
              transparent
              opacity={0.52}
            />
            {showLabels && (
              <Html
                position={[x, z + 1.2, hill.surfaceHalfWidth + 0.6]}
                center
                distanceFactor={55}
                style={{ pointerEvents: 'none' }}
              >
                <span className="scene-label">{x} m</span>
              </Html>
            )}
          </group>
        )
      })}

      <mesh
        position={[rampCentre[0], rampCentre[1], 0]}
        rotation={[0, 0, rampAngle]}
        receiveShadow
        castShadow
      >
        <boxGeometry args={[rampLength, 0.6, 8.5]} />
        <meshStandardMaterial color="#edf7f8" roughness={0.92} />
      </mesh>
      {[-1.25, 1.25].map((offset) => (
        <Line
          key={offset}
          points={[
            [rampStart[0], rampStart[1] + 0.38, offset],
            [rampEnd[0], rampEnd[1] + 0.38, offset],
          ]}
          color="#557985"
          lineWidth={1.2}
          transparent
          opacity={0.9}
        />
      ))}
      <mesh position={[-0.55, 0.82, 0]} receiveShadow castShadow>
        <boxGeometry args={[2.8, 0.5, 8.5]} />
        <meshStandardMaterial color="#d7e9ed" roughness={0.84} />
      </mesh>
    </group>
  )
}
