import { Line, OrbitControls } from '@react-three/drei'
import { Canvas, useThree } from '@react-three/fiber'
import { Suspense, useEffect, useMemo, useRef } from 'react'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import type { HillData, SimulationResult } from '../simulation/model-types'
import { stateAtTime } from '../simulation/ssm'
import { Jumper } from './Jumper'
import { LandingHill } from './LandingHill'
import { WindIndicators } from './WindIndicators'

export type CameraPreset = 'perspective' | 'side' | 'top' | 'landing'

interface CameraCommand {
  preset: CameraPreset
  nonce: number
}

interface JumpSceneProps {
  baselineResult: SimulationResult
  result: SimulationResult
  hill: HillData
  time: number
  showBaseline: boolean
  showAxes: boolean
  showWind: boolean
  showLabels: boolean
  cameraCommand: CameraCommand
}

const CAMERA_POSITIONS: Record<CameraPreset, [number, number, number]> = {
  perspective: [82, 38, 135],
  side: [90, -40, 145],
  top: [90, 145, 0.01],
  landing: [195, -85, 55],
}

const CAMERA_TARGET: [number, number, number] = [72, -38, 0]

function CameraRig({ command }: { command: CameraCommand }) {
  const controls = useRef<OrbitControlsImpl>(null)
  const { camera } = useThree()
  useEffect(() => {
    const position = CAMERA_POSITIONS[command.preset]
    camera.position.set(...position)
    camera.up.set(0, 1, 0)
    controls.current?.target.set(...CAMERA_TARGET)
    controls.current?.update()
  }, [camera, command])
  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping
      dampingFactor={0.07}
      rotateSpeed={0.72}
      zoomSpeed={0.8}
      panSpeed={0.75}
      minDistance={20}
      maxDistance={330}
      maxPolarAngle={Math.PI * 0.94}
      zoomToCursor
    />
  )
}

function SceneContent({
  baselineResult,
  result,
  hill,
  time,
  showBaseline,
  showAxes,
  showWind,
  showLabels,
  cameraCommand,
}: JumpSceneProps) {
  const baselinePoints = useMemo(
    () => baselineResult.states.map((state) => [state[0]!, state[2]!, state[1]!] as const),
    [baselineResult],
  )
  const resultPoints = useMemo(
    () => result.states.map((state) => [state[0]!, state[2]!, state[1]!] as const),
    [result],
  )
  const currentState = stateAtTime(result, time)
  const nextState = stateAtTime(result, time + 0.04)
  const sampleInterval = Math.max((result.times[1] ?? 0.04) - (result.times[0] ?? 0), 1e-6)
  const completedIndex = Math.min(Math.floor(time / sampleInterval), resultPoints.length - 1)
  const travelledPoints: Array<[number, number, number]> = resultPoints
    .slice(0, completedIndex + 1)
    .map((point) => [...point])
  const currentPoint: [number, number, number] = [currentState[0]!, currentState[2]!, currentState[1]!]
  const travelledEnd = travelledPoints[travelledPoints.length - 1]
  if (
    travelledEnd &&
    Math.hypot(
      travelledEnd[0] - currentPoint[0],
      travelledEnd[1] - currentPoint[1],
      travelledEnd[2] - currentPoint[2],
    ) > 1e-4
  ) {
    travelledPoints.push(currentPoint)
  }
  // Wind controls are constant over the current fitted horizon. Body controls
  // vary with time, but WindIndicators only reads the first twelve values.
  const currentControl = result.controls[0]!
  const endpoint = result.states[result.states.length - 1]!

  return (
    <>
      <color attach="background" args={['#08131b']} />
      <fog attach="fog" args={['#08131b', 150, 340]} />
      <ambientLight intensity={0.85} color="#bad9e1" />
      <hemisphereLight args={['#9ed6df', '#10252d', 1.1]} />
      <directionalLight
        position={[50, 85, 50]}
        intensity={2.1}
        color="#f1fbff"
        castShadow
        shadow-mapSize={[1024, 1024]}
      />
      <LandingHill hill={hill} showLabels={showLabels} />
      {showBaseline && (
        <Line
          points={baselinePoints}
          color="#36d6ee"
          lineWidth={1.3}
          dashed
          dashSize={1.1}
          gapSize={0.9}
          transparent
          opacity={0.44}
        />
      )}
      <Line
        points={resultPoints}
        color="#ffc49a"
        lineWidth={1.45}
        dashed
        dashSize={1.1}
        gapSize={0.8}
        transparent
        opacity={0.54}
      />
      {travelledPoints.length >= 2 && (
        <Line points={travelledPoints} color="#ff7438" lineWidth={2.1} />
      )}
      <mesh position={[endpoint[0]!, endpoint[2]! + 0.35, endpoint[1]!]}>
        <sphereGeometry args={[0.42, 18, 12]} />
        <meshStandardMaterial
          color="#ffc49a"
          emissive="#7a3212"
          emissiveIntensity={0.25}
          transparent
          opacity={0.62}
        />
      </mesh>
      <Jumper state={currentState} nextState={nextState} />
      {showWind && <WindIndicators control={currentControl} hill={hill} />}
      {showAxes && <axesHelper args={[18]} position={[0, 0.8, 0]} />}
      <CameraRig command={cameraCommand} />
    </>
  )
}

export function JumpScene(props: JumpSceneProps) {
  return (
    <Canvas
      camera={{ position: CAMERA_POSITIONS.perspective, fov: 46, near: 0.1, far: 700 }}
      dpr={[1, 1.75]}
      gl={{ antialias: true, powerPreference: 'high-performance' }}
      frameloop="demand"
      shadows
    >
      <Suspense fallback={null}>
        <SceneContent {...props} />
      </Suspense>
    </Canvas>
  )
}
