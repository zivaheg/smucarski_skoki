import { useMemo } from 'react'
import type { Vector } from '../simulation/model-types'

interface JumperProps {
  state: Vector
  nextState: Vector
}

export function Jumper({ state, nextState }: JumperProps) {
  const pitch = useMemo(() => {
    const dx = nextState[0]! - state[0]!
    const dz = nextState[2]! - state[2]!
    return Math.atan2(dz, Math.max(Math.abs(dx), 1e-6))
  }, [nextState, state])
  const yaw = Math.atan2(nextState[1]! - state[1]!, Math.max(nextState[0]! - state[0]!, 1e-6))

  return (
    <group
      position={[state[0]!, state[2]! + 0.35, state[1]!]}
      rotation={[0, -yaw, pitch]}
      scale={1.35}
    >
      <mesh castShadow position={[0, -0.25, -0.32]} rotation={[0, 0.075, 0]}>
        <boxGeometry args={[3.8, 0.07, 0.14]} />
        <meshStandardMaterial color="#f7fbfc" metalness={0.16} roughness={0.3} />
      </mesh>
      <mesh castShadow position={[0, -0.25, 0.32]} rotation={[0, -0.075, 0]}>
        <boxGeometry args={[3.8, 0.07, 0.14]} />
        <meshStandardMaterial color="#f7fbfc" metalness={0.16} roughness={0.3} />
      </mesh>
      <mesh castShadow position={[-0.4, 0.02, -0.19]} rotation={[0.12, 0, -0.52]}>
        <capsuleGeometry args={[0.11, 0.62, 5, 10]} />
        <meshStandardMaterial color="#263843" roughness={0.62} />
      </mesh>
      <mesh castShadow position={[-0.4, 0.02, 0.19]} rotation={[-0.12, 0, -0.52]}>
        <capsuleGeometry args={[0.11, 0.62, 5, 10]} />
        <meshStandardMaterial color="#263843" roughness={0.62} />
      </mesh>
      <mesh castShadow position={[0.08, 0.62, 0]} rotation={[0, 0, -0.2]}>
        <capsuleGeometry args={[0.27, 0.9, 7, 14]} />
        <meshStandardMaterial color="#f05f38" roughness={0.48} />
      </mesh>
      <mesh castShadow position={[-0.03, 0.68, -0.28]} rotation={[0.06, 0, -0.2]}>
        <boxGeometry args={[0.52, 0.62, 0.08]} />
        <meshStandardMaterial color="#f3f8f9" roughness={0.55} />
      </mesh>
      <mesh castShadow position={[0.33, 1.34, 0]}>
        <sphereGeometry args={[0.31, 18, 14]} />
        <meshStandardMaterial color="#f5c84d" roughness={0.34} metalness={0.08} />
      </mesh>
      <mesh castShadow position={[0.55, 1.34, -0.02]} rotation={[0, 0, -0.06]}>
        <sphereGeometry args={[0.22, 16, 10, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshStandardMaterial color="#17313d" roughness={0.2} metalness={0.35} />
      </mesh>
      <mesh castShadow position={[-0.05, 0.54, -0.58]} rotation={[0.28, 0, -0.72]}>
        <capsuleGeometry args={[0.1, 0.95, 5, 10]} />
        <meshStandardMaterial color="#f05f38" roughness={0.5} />
      </mesh>
      <mesh castShadow position={[-0.05, 0.54, 0.58]} rotation={[-0.28, 0, -0.72]}>
        <capsuleGeometry args={[0.1, 0.95, 5, 10]} />
        <meshStandardMaterial color="#f05f38" roughness={0.5} />
      </mesh>
    </group>
  )
}
