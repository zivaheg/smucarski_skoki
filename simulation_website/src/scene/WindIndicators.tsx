import { Html } from '@react-three/drei'
import { useEffect, useMemo } from 'react'
import { ArrowHelper, Color, Vector3 } from 'three'
import type { HillData, Vector } from '../simulation/model-types'
import { hillZAt } from '../simulation/metrics'

interface WindIndicatorsProps {
  control: Vector
  hill: HillData
}

const ZONES = [
  { x: 25, name: 'takeoff', label: 'Takeoff wind', color: '#45c6df' },
  { x: 90, name: 'middle', label: 'Mid-flight wind', color: '#b79cff' },
  { x: 150, name: 'landing', label: 'Landing wind', color: '#f2b85b' },
]

export function WindIndicators({ control, hill }: WindIndicatorsProps) {
  const arrows = useMemo(
    () =>
      ZONES.map((zone, zoneIndex) => {
        const speed = control[zoneIndex] ?? 0
        const tangent = control[3 + zoneIndex] ?? 0
        const cross = control[6 + zoneIndex] ?? 0
        const turbulence = control[9 + zoneIndex] ?? 0
        const direction = new Vector3(tangent || speed || 0.1, 0.08 + turbulence * 0.25, cross)
        direction.normalize()
        const ground = hillZAt(hill, zone.x) ?? 0
        const length = Math.max(3, Math.min(9, 3.5 + Math.abs(speed) * 2.3))
        const lateralPosition = -hill.surfaceHalfWidth - 3.5
        return {
          helper: new ArrowHelper(
            direction,
            new Vector3(zone.x, ground + 6.5, lateralPosition),
            length,
            new Color(zone.color),
            1.25,
            0.65,
          ),
          zone,
          ground,
          lateralPosition,
          speed,
          turbulence,
        }
      }),
    [control, hill],
  )

  useEffect(
    () => () => {
      arrows.forEach(({ helper }) => {
        helper.line.geometry.dispose()
        if (Array.isArray(helper.line.material)) {
          helper.line.material.forEach((material) => material.dispose())
        } else {
          helper.line.material.dispose()
        }
        helper.cone.geometry.dispose()
        if (Array.isArray(helper.cone.material)) {
          helper.cone.material.forEach((material) => material.dispose())
        } else {
          helper.cone.material.dispose()
        }
      })
    },
    [arrows],
  )

  return (
    <group>
      {arrows.map(({ helper, zone, ground, lateralPosition, speed, turbulence }) => (
        <group key={zone.name}>
          <primitive object={helper} />
          <Html
            position={[zone.x, ground + 9, lateralPosition]}
            center
            distanceFactor={55}
            style={{ pointerEvents: 'none' }}
          >
            <span className="wind-marker" style={{ '--wind-color': zone.color } as React.CSSProperties}>
              <strong>{zone.label}</strong>
              <small>{speed.toFixed(2)} m/s · turbulence {turbulence.toFixed(2)}</small>
            </span>
          </Html>
        </group>
      ))}
    </group>
  )
}
