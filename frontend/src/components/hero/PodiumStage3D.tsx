import { ContactShadows, Environment, Html, Lightformer, MeshReflectorMaterial } from '@react-three/drei'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'

import { UNKNOWN_COMPOUND_COLOR } from '@/lib/teams'

import { F1Car3D } from './F1Car3D'
import { RANKS, SLOTS, type StageCar, type StageProps } from './slots'

const FOV = 27
const CAM_Y = 2.3
const LOOK = new THREE.Vector3(0, 1.05, -1.6)
const CAR_SCALE = 1.15
const REVEAL_S = 1.3
const LASER = '#00d2be'
const GHOST_ACCENT = UNKNOWN_COMPOUND_COLOR

const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches

/** Camera: pulled back on narrow stages so three cars still fit, and eased toward the pointer for parallax. */
function Rig() {
  const { camera, size, pointer } = useThree()
  const reduced = useMemo(() => reducedMotion(), [])
  useFrame((_, dt) => {
    const aspect = size.width / size.height
    const dist = Math.max(9.6, 15 / aspect, 4300 / size.height)
    camera.position.x = reduced ? 0 : THREE.MathUtils.damp(camera.position.x, pointer.x * 0.5, 3, dt)
    camera.position.y = CAM_Y + Math.max(0, (dist - 9.6) * 0.12)
    camera.position.z = dist
    camera.lookAt(LOOK)
  })
  return null
}

function useRadialAlpha() {
  return useMemo(() => {
    const canvas = document.createElement('canvas')
    canvas.width = canvas.height = 256
    const ctx = canvas.getContext('2d')!
    const fade = ctx.createRadialGradient(128, 128, 8, 128, 128, 128)
    fade.addColorStop(0, '#fff')
    fade.addColorStop(0.5, '#c8c8c8')
    fade.addColorStop(1, '#000')
    ctx.fillStyle = fade
    ctx.fillRect(0, 0, 256, 256)
    return new THREE.CanvasTexture(canvas)
  }, [])
}

/** Studio HDRI from procedural light panels (nothing fetched over the network),
 *  a reflective floor that fades into the page, and converging neon ground lines. */
function Studio() {
  const alpha = useRadialAlpha()
  const lines = useMemo(() => [-7, -4.2, -1.9, 1.9, 4.2, 7], [])
  return (
    <>
      <ambientLight intensity={0.12} />
      <directionalLight position={[4, 7, 5]} intensity={1.8} color="#e8eef6" />
      {/* rim lights from behind: a silver edge and a laser edge, so the carbon reads as a surface */}
      <directionalLight position={[5, 3.5, -8]} intensity={2.6} color="#dfe8f2" />
      <directionalLight position={[-6, 3, -6]} intensity={1.1} color={LASER} />
      <Environment resolution={256} frames={1}>
        <Lightformer form="rect" intensity={2.4} color="#dfe8f2" position={[0, 5, 2]} rotation-x={Math.PI / 2} scale={[14, 5, 1]} />
        <Lightformer form="rect" intensity={1.6} color={LASER} position={[-6, 1.5, -1]} rotation-y={Math.PI / 2} scale={[8, 2.5, 1]} />
        <Lightformer form="rect" intensity={1.1} color="#e2e8f0" position={[6, 2, 2]} rotation-y={-Math.PI / 2} scale={[8, 2.5, 1]} />
        <Lightformer form="ring" intensity={0.9} color="#00a19b" position={[0, 0.4, -9]} scale={6} />
      </Environment>

      <mesh rotation-x={-Math.PI / 2} position={[0, 0, -4]}>
        <planeGeometry args={[36, 36]} />
        <MeshReflectorMaterial
          color="#0a0e14"
          metalness={0.15}
          roughness={0.85}
          mirror={0.85}
          resolution={512}
          blur={[240, 70]}
          mixBlur={1}
          mixStrength={14}
          depthScale={0.9}
          minDepthThreshold={0.4}
          maxDepthThreshold={1.3}
          transparent
          alphaMap={alpha}
        />
      </mesh>
      {lines.map((x) => (
        <mesh key={x} position={[x, 0.004, -5]} rotation-x={-Math.PI / 2} rotation-z={-x * 0.012}>
          <planeGeometry args={[0.035, 30]} />
          <meshBasicMaterial color={LASER} transparent opacity={0.4} toneMapped={false} />
        </mesh>
      ))}
      <ContactShadows position={[0, 0.006, -2.5]} opacity={0.7} scale={22} blur={2.4} far={2.2} resolution={256} />
    </>
  )
}

function PodiumCar({ rank, driver, hovered, onHover, onSelect, renderHud }: StageCar & Pick<StageProps, 'hovered' | 'onHover' | 'onSelect' | 'renderHud'>) {
  const slot = SLOTS[rank]
  const group = useRef<THREE.Group>(null)
  const ring = useRef<THREE.MeshBasicMaterial>(null)
  const born = useRef<number | null>(null)
  const lift = useRef(0)
  const { camera, size, clock } = useThree()
  const reduced = useMemo(() => reducedMotion(), [])
  const isHovered = hovered === rank

  useFrame((_, dt) => {
    const g = group.current
    if (!g) return
    if (born.current === null) born.current = clock.elapsedTime
    const p = reduced ? 1 : THREE.MathUtils.clamp((clock.elapsedTime - born.current - slot.delay) / REVEAL_S, 0, 1)
    const eased = 1 - Math.pow(1 - p, 4)

    // land the car on its screen slot whatever the stage aspect, then restore the brief's size ratio
    const d = camera.position.z - slot.z
    const d1 = camera.position.z - SLOTS[1].z
    const halfWidth = Math.tan(THREE.MathUtils.degToRad(FOV / 2)) * d * (size.width / size.height)
    lift.current = THREE.MathUtils.damp(lift.current, isHovered ? 0.07 : 0, 8, dt)
    const bob = reduced ? 0 : Math.sin(clock.elapsedTime * 0.8 + rank) * 0.012

    g.position.set(slot.ndc * halfWidth * 0.9, lift.current + bob, slot.z + (1 - eased) * -9)
    g.scale.setScalar(CAR_SCALE * slot.scale * (d / d1))
    g.rotation.set(0.05, slot.yaw + (isHovered ? 0.06 : 0), 0) // nose dipped forward, turned in toward the winner
    if (ring.current) ring.current.opacity = THREE.MathUtils.damp(ring.current.opacity, (isHovered ? 1 : 0.7) * slot.key * (driver ? 1 : 0.25), 6, dt)
  })

  const interactive = driver
    ? {
        onPointerOver: (e: { stopPropagation: () => void }) => {
          e.stopPropagation()
          onHover(rank)
          document.body.style.cursor = 'pointer'
        },
        onPointerOut: () => {
          onHover(null)
          document.body.style.cursor = ''
        },
        onClick: (e: { stopPropagation: () => void }) => {
          e.stopPropagation()
          onSelect(driver)
        },
      }
    : {}

  return (
    <group ref={group} {...interactive}>
      <F1Car3D accent={driver?.accent ?? GHOST_ACCENT} compound={driver?.compound ?? null} dim={driver ? slot.key : 0.3} />
      {/* ground light: an ellipse just larger than the car's footprint, plus underglow */}
      <mesh rotation-x={-Math.PI / 2} position-y={0.008} scale={[1, 1.9, 1]}>
        <ringGeometry args={[1.5, 1.56, 72]} />
        <meshBasicMaterial ref={ring} color={LASER} transparent opacity={0} toneMapped={false} />
      </mesh>
      <pointLight color={LASER} intensity={(driver ? 7 : 1.5) * slot.key} distance={4.5} decay={2} position={[0, 0.12, 0]} />
      {driver && (
        <Html position={[0, 1.42, 0.3]} zIndexRange={[30 + rank * 10, 0]}>
          <div style={{ transform: 'translate(-50%, -100%)' }}>{renderHud(driver, rank)}</div>
        </Html>
      )}
    </group>
  )
}

function Cars(props: StageProps) {
  return (
    <>
      {RANKS.map((rank) => (
        <PodiumCar
          key={rank}
          rank={rank}
          driver={props.cars.find((c) => c.rank === rank)?.driver ?? null}
          hovered={props.hovered}
          onHover={props.onHover}
          onSelect={props.onSelect}
          renderHud={props.renderHud}
        />
      ))}
    </>
  )
}

export default function PodiumStage3D(props: StageProps) {
  const wrap = useRef<HTMLDivElement>(null)
  const [inView, setInView] = useState(true)

  // stop rendering while the stage is scrolled out of view (mobile: the deck sits below it)
  useEffect(() => {
    const el = wrap.current
    if (!el) return
    const observer = new IntersectionObserver(([entry]) => setInView(entry.isIntersecting))
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  return (
    <div ref={wrap} className="absolute inset-0">
      <Canvas
        dpr={[1, 1.75]}
        camera={{ fov: FOV, position: [0, CAM_Y, 9.6], near: 0.5, far: 60 }}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        frameloop={inView ? 'always' : 'never'}
      >
        <fog attach="fog" args={['#05070a', 13, 30]} />
        <Rig />
        <Suspense fallback={null}>
          <Studio />
          {/* keyed on the race: a new selection remounts the cars and replays the reveal */}
          <Cars key={props.raceKey} {...props} />
        </Suspense>
      </Canvas>
    </div>
  )
}

