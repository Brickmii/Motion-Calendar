# Motion Calendar Simulation — Task 2: Particle Formation & SU Group Emergence

## Overview

Extend the existing Blender simulation (Motion-Calendar-Sim.blend) to show how particles emerge from the Motion Calendar's six functions. The simulation currently shows: Heat dot → Polarity line (+/-) → Existence dot (stationary, spinning at midpoint) → Cubic expansion. This task adds the next phase: particle formation through SU group emergence.

---

## Theoretical Foundation

### The Chain of Emergence

Each motion function builds on the last. Particles arise from the interaction of ALL six:

1. **Heat (κ)** — Pure magnitude. A particle's heat value = its "mass-energy." Heat only accumulates, never subtracts. The thermal quantum K = 4/φ² ≈ 1.528.

2. **Polarity (+/-)** — Opposition. Every particle carries a polarity state. Polarity is conserved in all interactions.

3. **Existence (E)** — Instantiation in time. The existence dot at the polarity midpoint is the first persistent structure. E(M,t) ∈ {0,1}.

4. **Righteousness (R)** — Relational evaluation across a 4-quadrant oppositional frame. R=0 is perfect alignment. The 4 quadrants come from the Quadratic (QUADRATIC = 4 in node_constants).

5. **Order (Q)** — Structural invariance. Stabilizes symmetric groups S(n) — permutation groups of motion tokens under Robinson arithmetic (Q axioms: identity, successor, addition, closure without induction).

6. **Movement** — 12 directional operators (6 self-frame + 6 universal-frame). Assigns trajectories. The emergence threshold 45/44 governs splits.

### How SU Groups Arise

This is the key theoretical claim being visualized:

- **Order** stabilizes **S(n)** — symmetric/permutation groups
- **Polarity** acting on S(n) forces conservation of +/- across all 4 Righteousness quadrants simultaneously
- This polarity constraint on the permutation groups produces **SU(n)** as subgroups
- The "Special" in Special Unitary = the polarity conservation (determinant = 1 condition)
- **SU(2)** and **SU(3)** emerge depending on how many quadrants are actively constrained

In physics terms:
- SU(2) = weak force gauge group (2 active quadrant pairs)
- SU(3) = strong force gauge group (3 active quadrant pairs → 8 generators = gluons)
- The Standard Model gauge structure is not imposed — it emerges from Order + Polarity + Righteousness

### The Emergence Threshold: 45/44

From node_constants.py:
```
45/44 = (4 × 11 + 1) / (4 × 11)
      = (Quadratic × Incomplete_Motion + 1) / (Quadratic × Incomplete_Motion)
```

Where:
- 4 = Quadratic (from Righteousness — 4 quadrants)
- 11 = 12 - 1 = Incomplete Motion (movement directions minus one)
- +1 = the completion that tips the system into a new motion function

This ratio governs ALL particle splits. The inverse 44/45 ≈ 0.9778 is the maximum entropic probability — the cap on randomness. The 1/45 gap is where structure lives.

---

## The Motion Space (Cube)

**CRITICAL**: The cube is NOT a container. It is a direct expression of Heat magnitude made spatial.

- The cube arises to EXACTLY support the magnitude of the motion within it
- More heat = larger cube
- If potential motion collapses, the cube CONTRACTS (this is already working in the sim)
- The cube IS the motion space — all space arises to support a motion
- Formally: the Color Cube / Chromatic Heat State Space from node_constants (3 axes: opponent chroma A [-1,+1], opponent chroma B [-1,+1], heat magnitude [0,+1])

---

## Particle Properties

Each particle in the simulation carries:

| Property | Source Function | Visual Representation |
|----------|---------------|----------------------|
| Heat magnitude (κ) | Heat | Size of particle (larger = more heat) |
| Polarity state per quadrant | Polarity × Righteousness | Color bands (4 colored segments, one per quadrant, color indicates +/-) |
| Symmetry group type | Order + Polarity | Shape (see below) |
| Trajectory | Movement | Direction of travel |
| Existence state | Existence | Visible (E=1) or absent (E=0) |

### Shape by Symmetry Group

| Group | Shape | Meaning |
|-------|-------|---------|
| S(1) / trivial | Sphere | Undifferentiated — no internal permutation structure |
| SU(2) | Torus / double-ring | 2-quadrant polarity constraint — weak interaction carrier |
| SU(3) | Triangular prism / 3-lobed | 3-quadrant polarity constraint — strong interaction carrier |
| S(n) pre-constraint | Icosahedron / complex polyhedron | Permutation group before polarity reduces it |

### Color by Polarity State

Each particle has 4 quadrant positions from Righteousness. Each quadrant carries a polarity (+/-):

| Quadrant | + Color | - Color |
|----------|---------|---------|
| Q1 | Red | Cyan |
| Q2 | Green | Magenta |
| Q3 | Blue | Yellow |
| Q4 | White | Black |

The particle mesh is divided into 4 segments (bands, faces, or quadrants of the surface), each colored by its polarity state in that quadrant. This makes the polarity configuration immediately visible.

---

## Particle Dynamics

### Splits (45/44 Emergence Threshold)

When a particle's heat exceeds an emergence threshold, it splits:

1. **Heat division**: Discrete, not a ratio.
   - Child A gets exactly **45 units** of heat
   - Child B gets exactly **44 units** of heat (45 - 1, the 1/45 structural gap)
   - The 1-unit difference IS the structural constraint — the gap where structure lives
   - Each child then accumulates heat independently
   - When a child's potential motion exceeds the **4 × 11 = 44 threshold**, it splits again
   - The cycle repeats: every split produces a 45 and a 44, every time

2. **Direction conservation**: Trajectories diverge symmetrically around parent's direction.
   - Parent traveling North → Child A goes NW, Child B goes NE
   - Parent traveling Up → Child A goes Up-Left, Child B goes Up-Right
   - The split angle should be visually clear (e.g., 45° divergence)

3. **Polarity conservation**: The +/- balance across ALL 4 quadrants must be preserved.
   - If parent is (+,+,-,+) across Q1-Q4, children's combined quadrant polarities must sum to the same
   - This constraint determines which SU subgroup the children belong to
   - NOT all splits are possible — only those that conserve polarity across all 4 quadrants

4. **Group inheritance**: The SU group type of children is determined by HOW polarity distributes across the quadrants during the split.

### Collisions

When two particles meet (occupy same spatial region):

1. **Heat conserved**: Total heat before = total heat after
2. **Direction conserved**: Total directional momentum preserved
3. **Polarity conserved**: Combined quadrant polarities preserved
4. **Group interaction**: Compatible symmetry groups may merge; incompatible ones deflect

### Cube Response

- After splits: cube stays same size (heat conserved, no new heat created)
- After collapses: if motion potential is lost, cube contracts
- The cube continuously sizes itself to exactly support the heat magnitude present

---

## Animation Sequence

### Phase 1: Existing (Already Implemented)
- Frame 1-30: Heat dot appears (spinning, glowing)
- Frame 31-60: Polarity line extends (+ and - ends)
- Frame 61-90: Existence dot appears at midpoint, remains stationary and spinning
- Frame 91-120: Cubic expansion to full motion space

### Phase 2: Order Emergence (NEW — Frames 121-180)
- Frame 121-140: Existence dot begins vibrating/pulsing — Order is stabilizing
- Frame 141-160: S(n) permutation structure becomes visible — the dot transforms from sphere to icosahedron/complex shape, showing internal structural symmetry
- Frame 160-180: The shape settles — Order has produced a stable S(n) group

### Phase 3: SU Group Emergence via Polarity (NEW — Frames 181-240)
- Frame 181-200: Polarity colors appear on the ordered structure — 4 quadrant colors emerge on the surface, showing Righteousness quadrants being activated
- Frame 200-220: The polarity constraint acts — the S(n) shape simplifies/reduces as polarity forces the Special Unitary condition. Icosahedron → Torus (SU(2)) or → Triangular (SU(3))
- Frame 220-240: First stable particle fully formed with shape (group) + color (polarity state)

### Phase 4: First Split (NEW — Frames 241-320)
- Frame 241-260: The particle's heat magnitude pulses — approaching emergence threshold
- Frame 261-280: 45/44 SPLIT — particle divides into two children
  - Diverging trajectories (symmetric around parent direction)
  - Heat visibly redistributed (one slightly larger than other)
  - Polarity colors redistribute while conserving across all 4 quadrants
  - Children may have different SU group types depending on polarity distribution
- Frame 280-320: Children travel along their trajectories, cube adjusts

### Phase 5: Cascade and Collisions (NEW — Frames 321-480)
- Frame 321-400: Additional splits occur as children reach their own thresholds
- Frame 400-480: First collision event
  - Two particles approach from different directions
  - Collision conserves heat + direction + polarity
  - Result: merged particle, deflected particles, or new split depending on group compatibility
- Multiple particles now populating the motion space
- Cube dynamically sizing to motion magnitude

---

## Technical Notes

### Constants to Use (from node_constants.py)

```python
PHI = (1 + math.sqrt(5)) / 2          # ≈ 1.618
K = 4 / (PHI ** 2)                     # ≈ 1.528 (thermal quantum)
EMERGENCE_THRESHOLD = 45 / 44          # ≈ 1.02273 (split ratio)
QUADRATIC = 4                           # Righteousness quadrants
MOVEMENT_DIRECTIONS = 12               # 6 self + 6 universal
THRESHOLD_HEAT = 1/PHI                 # ≈ 0.618
THRESHOLD_POLARITY = 1/PHI**2          # ≈ 0.382
THRESHOLD_EXISTENCE = 1/PHI**3         # ≈ 0.236
THRESHOLD_RIGHTEOUSNESS = 1/PHI**4     # ≈ 0.146
THRESHOLD_ORDER = 1/PHI**5             # ≈ 0.090
THRESHOLD_MOVEMENT = 1/PHI**6          # ≈ 0.056
```

### Blender Implementation Guidelines

- Use Blender's Python API (bpy)
- Keyframe all animations
- Materials: Use Emission shaders for glowing particles, varying intensity with heat
- Particle shapes: Use mesh primitives (UV Sphere, Torus, triangular prism via custom mesh)
- Polarity colors: Assign vertex groups or face materials for the 4 quadrant segments
- Cube: Wireframe material, scale keyframed to heat magnitude
- Camera: Should pull back as particle count increases to keep all in frame
- Lighting: Minimal — particles self-illuminate via emission

### File Structure

- All changes should be in the Blender Python script embedded in Motion-Calendar-Sim.blend
- The script should be self-contained and runnable
- Constants from node_constants.py should be embedded directly (don't import the module)

---

## Acceptance Criteria

1. ✅ Existing Phase 1 animation unchanged (Heat → Polarity → Existence → Cube)
2. ✅ Order emergence visible: particle transforms from simple sphere to complex polyhedron
3. ✅ SU group emergence visible: polarity colors appear, shape reduces as constraint acts
4. ✅ Particle carries visual properties: size=heat, shape=group, color=polarity per quadrant
5. ✅ First 45/44 split: symmetric trajectory divergence, heat conserved, polarity conserved across all 4 quadrants
6. ✅ Multiple splits cascade
7. ✅ At least one collision event with conservation laws respected
8. ✅ Cube dynamically sizes to total heat magnitude
9. ✅ Cube contracts when motion collapses
10. ✅ Animation is smooth and visually clear

---

## Reference Materials

- Full Motion Calendar papers: Universe_In_Motion/ (6 papers + intro)
- Node constants: node_constants.py (all mathematical constants)
- Existing sim: Motion-Calendar-Sim.blend in the Motion-Calendar repo
