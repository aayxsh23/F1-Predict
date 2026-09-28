import { useMemo } from 'react'
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'

import { COMPOUNDS, UNKNOWN_COMPOUND_COLOR } from '@/lib/teams'

type V3 = [number, number, number]

/* A procedural single-seater, authored from primitives: no licensed model, no team
   livery (PRODUCT.md). Nose points +Z, metres, resting on y = 0. The only team
   colour is a trim accent; the tyre sidewall ring shows the starting compound. */

/** A rounded box whose rear face is drawn in toward the centreline: a linear
 *  front-to-back taper, the coke-bottle of a sidepod or engine cover. */
function taperedBox(w: number, h: number, l: number, rearW: number, rearH: number, radius: number) {
  const g = new RoundedBoxGeometry(w, h, l, 4, radius)
  const p = g.attributes.position
  for (let i = 0; i < p.count; i++) {
    const t = (p.getZ(i) + l / 2) / l // 0 at the rear, 1 at the nose
    p.setX(i, p.getX(i) * (rearW + (1 - rearW) * t))
    p.setY(i, p.getY(i) * (rearH + (1 - rearH) * t))
  }
  g.computeVertexNormals()
  return g
}

/** A top-down outline of [x, z] points extruded into a slab of thickness `thick`. */
function slab(points: Array<[number, number]>, thick: number) {
  const shape = new THREE.Shape(points.map(([x, z]) => new THREE.Vector2(x, z)))
  const g = new THREE.ExtrudeGeometry(shape, {
    depth: thick,
    bevelEnabled: true,
    bevelThickness: 0.008,
    bevelSize: 0.008,
    bevelSegments: 2,
    curveSegments: 8,
  })
  g.rotateX(Math.PI / 2) // outline y -> +z, extrusion -> down
  g.translate(0, thick, 0)
  return g
}

/** A curved tube through control points, for the halo. */
function tube(points: V3[], radius: number) {
  return new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p))), 24, radius, 8, false)
}

function Rod({ from, to, r = 0.011, material }: { from: V3; to: V3; r?: number; material: THREE.Material }) {
  const { position, quaternion, length } = useMemo(() => {
    const a = new THREE.Vector3(...from)
    const b = new THREE.Vector3(...to)
    const dir = b.clone().sub(a)
    const len = dir.length()
    return {
      position: a.clone().add(b).multiplyScalar(0.5),
      quaternion: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize()),
      length: len,
    }
  }, [from, to])
  return (
    <mesh position={position} quaternion={quaternion} material={material}>
      <cylinderGeometry args={[r, r, length, 8]} />
    </mesh>
  )
}

interface Materials {
  carbon: THREE.Material
  matte: THREE.Material
  silver: THREE.Material
  titanium: THREE.Material
  rubber: THREE.Material
  accent: THREE.Material
  laser: THREE.Material
  visor: THREE.Material
  compound: THREE.Material
}

function Wheel({ x, z, r, w, m }: { x: number; z: number; r: number; w: number; m: Materials }) {
  const side = Math.sign(x)
  return (
    <group position={[x, r, z]}>
      <mesh rotation-z={Math.PI / 2} material={m.rubber}>
        <cylinderGeometry args={[r, r, w, 48]} />
      </mesh>
      {/* outer face: carbon wheel cover, compound-coloured sidewall band, hub */}
      <group position-x={side * (w / 2 + 0.004)} rotation-y={(side * Math.PI) / 2}>
        <mesh material={m.matte}>
          <circleGeometry args={[r * 0.7, 40]} />
        </mesh>
        <mesh position-z={0.002} material={m.compound}>
          <ringGeometry args={[r * 0.8, r * 0.9, 56]} />
        </mesh>
        <mesh position-z={0.004} material={m.laser}>
          <ringGeometry args={[r * 0.07, r * 0.13, 24]} />
        </mesh>
      </group>
    </group>
  )
}

export function F1Car3D({ accent, compound, dim = 1 }: { accent: string; compound: string | null; dim?: number }) {
  const compoundColor = compound && COMPOUNDS[compound] ? COMPOUNDS[compound].color : UNKNOWN_COMPOUND_COLOR

  const m = useMemo<Materials>(
    () => ({
      carbon: new THREE.MeshPhysicalMaterial({ color: '#0e131b', metalness: 0.6, roughness: 0.32, clearcoat: 1, clearcoatRoughness: 0.05, envMapIntensity: 1.3 * dim }),
      matte: new THREE.MeshStandardMaterial({ color: '#0a0d12', metalness: 0.25, roughness: 0.62, envMapIntensity: dim }),
      silver: new THREE.MeshStandardMaterial({ color: '#c8d2de', metalness: 0.95, roughness: 0.22, envMapIntensity: dim }),
      titanium: new THREE.MeshStandardMaterial({ color: '#8592a3', metalness: 0.9, roughness: 0.33, envMapIntensity: dim }),
      rubber: new THREE.MeshStandardMaterial({ color: '#0e1013', roughness: 0.93 }),
      accent: new THREE.MeshStandardMaterial({ color: accent, emissive: accent, emissiveIntensity: 0.12 * dim, metalness: 0.4, roughness: 0.3 }),
      laser: new THREE.MeshBasicMaterial({ color: '#00d2be', toneMapped: false }),
      visor: new THREE.MeshPhysicalMaterial({ color: '#05070a', metalness: 0.9, roughness: 0.06, clearcoat: 1 }),
      compound: new THREE.MeshBasicMaterial({ color: compoundColor }),
    }),
    [accent, compoundColor, dim],
  )

  const g = useMemo(
    () => ({
      floor: slab([[-0.74, 1.45], [0.74, 1.45], [0.78, 0.3], [0.84, -0.9], [0.66, -2.2], [-0.66, -2.2], [-0.84, -0.9], [-0.78, 0.3]], 0.03),
      sidepod: taperedBox(0.46, 0.4, 1.9, 0.42, 0.55, 0.07),
      cover: taperedBox(0.4, 0.46, 1.95, 0.14, 0.2, 0.06),
      airbox: taperedBox(0.32, 0.3, 0.5, 0.8, 0.9, 0.05),
      haloL: tube([[-0.28, 0.64, -0.3], [-0.31, 0.8, -0.05], [-0.2, 0.88, 0.38], [-0.04, 0.86, 0.68]], 0.018),
      haloR: tube([[0.28, 0.64, -0.3], [0.31, 0.8, -0.05], [0.2, 0.88, 0.38], [0.04, 0.86, 0.68]], 0.018),
    }),
    [],
  )

  return (
    <group>
      {/* floor and diffuser */}
      <mesh geometry={g.floor} position-y={0.06} material={m.matte} />
      <mesh position={[0, 0.13, -2.3]} rotation-x={-0.25} material={m.matte}>
        <boxGeometry args={[0.92, 0.02, 0.6]} />
      </mesh>

      {/* nose cone, monocoque and cockpit */}
      {/* rotated +90deg about X the cylinder's narrow end points at +Z, and its local Z is world height */}
      <mesh position={[0, 0.3, 1.9]} rotation-x={Math.PI / 2 + 0.05} scale={[1, 1, 0.78]} material={m.carbon}>
        <cylinderGeometry args={[0.055, 0.17, 1.55, 28]} />
      </mesh>
      <mesh position={[0, 0.47, 0.35]} rotation-x={Math.PI / 2} scale={[1, 1, 0.82]} material={m.carbon}>
        <capsuleGeometry args={[0.29, 1.5, 8, 24]} />
      </mesh>
      <mesh position={[0, 0.7, 0.12]} material={m.visor}>
        <boxGeometry args={[0.34, 0.1, 0.62]} />
      </mesh>
      <mesh position={[0, 0.73, 0.03]} material={m.accent}>
        <sphereGeometry args={[0.125, 24, 18]} />
      </mesh>
      <mesh position={[0, 0.735, 0.13]} scale={[0.95, 0.42, 0.5]} material={m.visor}>
        <sphereGeometry args={[0.11, 20, 14]} />
      </mesh>

      {/* halo */}
      <mesh geometry={g.haloL} material={m.titanium} />
      <mesh geometry={g.haloR} material={m.titanium} />
      <Rod from={[0, 0.86, 0.68]} to={[0, 0.6, 0.9]} r={0.02} material={m.titanium} />

      {/* sidepods, inlets, engine cover, airbox, shark fin */}
      {[-1, 1].map((s) => (
        <group key={s}>
          <mesh geometry={g.sidepod} position={[s * 0.5, 0.36, -0.3]} material={m.carbon} />
          <mesh position={[s * 0.5, 0.4, 0.66]} material={m.visor}>
            <boxGeometry args={[0.34, 0.26, 0.02]} />
          </mesh>
        </group>
      ))}
      <mesh geometry={g.cover} position={[0, 0.6, -1.25]} material={m.carbon} />
      <mesh geometry={g.airbox} position={[0, 0.92, -0.32]} material={m.carbon} />
      <mesh position={[0, 0.92, -0.06]} material={m.visor}>
        <planeGeometry args={[0.24, 0.17]} />
      </mesh>
      <mesh position={[0, 0.86, -1.3]} material={m.carbon}>
        <boxGeometry args={[0.02, 0.32, 1.1]} />
      </mesh>
      <mesh position={[0, 0.84, -0.9]} material={m.accent}>
        <boxGeometry args={[0.03, 0.012, 1.3]} />
      </mesh>

      {/* front wing: main plane, two flaps, endplates, laser-lit leading edge */}
      <mesh position={[0, 0.075, 2.62]} material={m.carbon}>
        <boxGeometry args={[1.9, 0.022, 0.42]} />
      </mesh>
      <mesh position={[0, 0.125, 2.55]} rotation-x={0.18} material={m.carbon}>
        <boxGeometry args={[1.66, 0.018, 0.26]} />
      </mesh>
      <mesh position={[0, 0.17, 2.5]} rotation-x={0.3} material={m.silver}>
        <boxGeometry args={[1.4, 0.016, 0.2]} />
      </mesh>
      <mesh position={[0, 0.078, 2.835]} material={m.laser}>
        <boxGeometry args={[1.9, 0.006, 0.012]} />
      </mesh>
      {[-1, 1].map((s) => (
        <group key={s}>
          <mesh position={[s * 0.95, 0.15, 2.58]} material={m.carbon}>
            <boxGeometry args={[0.02, 0.24, 0.6]} />
          </mesh>
          <mesh position={[s * 0.95, 0.27, 2.58]} material={m.accent}>
            <boxGeometry args={[0.026, 0.02, 0.6]} />
          </mesh>
          <mesh position={[s * 0.09, 0.17, 2.4]} material={m.matte}>
            <boxGeometry args={[0.03, 0.16, 0.34]} />
          </mesh>
        </group>
      ))}

      {/* rear wing, beam wing, pylon */}
      <mesh position={[0, 0.98, -2.45]} rotation-x={-0.12} material={m.carbon}>
        <boxGeometry args={[1.02, 0.03, 0.36]} />
      </mesh>
      <mesh position={[0, 1.07, -2.5]} rotation-x={-0.35} material={m.silver}>
        <boxGeometry args={[1.02, 0.022, 0.24]} />
      </mesh>
      <mesh position={[0, 0.66, -2.42]} material={m.matte}>
        <boxGeometry args={[0.84, 0.02, 0.2]} />
      </mesh>
      <mesh position={[0, 0.8, -2.3]} material={m.matte}>
        <boxGeometry args={[0.04, 0.4, 0.16]} />
      </mesh>
      {[-1, 1].map((s) => (
        <group key={s}>
          <mesh position={[s * 0.51, 0.9, -2.48]} material={m.carbon}>
            <boxGeometry args={[0.025, 0.44, 0.64]} />
          </mesh>
          <mesh position={[s * 0.51, 1.12, -2.48]} material={m.accent}>
            <boxGeometry args={[0.03, 0.02, 0.64]} />
          </mesh>
        </group>
      ))}

      {/* wheels and suspension */}
      {[-1, 1].map((s) => (
        <group key={s}>
          <Wheel x={s * 0.86} z={1.75} r={0.35} w={0.3} m={m} />
          <Wheel x={s * 0.82} z={-1.65} r={0.36} w={0.4} m={m} />
          <Rod from={[s * 0.22, 0.44, 1.62]} to={[s * 0.7, 0.42, 1.75]} material={m.titanium} />
          <Rod from={[s * 0.24, 0.17, 1.5]} to={[s * 0.7, 0.24, 1.75]} material={m.titanium} />
          <Rod from={[s * 0.7, 0.4, 1.75]} to={[s * 0.16, 0.6, 1.4]} r={0.014} material={m.silver} />
          <Rod from={[s * 0.24, 0.42, -1.5]} to={[s * 0.64, 0.42, -1.65]} material={m.titanium} />
          <Rod from={[s * 0.26, 0.16, -1.8]} to={[s * 0.64, 0.24, -1.65]} material={m.titanium} />
          <Rod from={[s * 0.64, 0.4, -1.65]} to={[s * 0.18, 0.62, -1.3]} r={0.014} material={m.silver} />
        </group>
      ))}
    </group>
  )
}
