# Gait Studio

Design a walking style for the robot interactively, check it against what the STS3215 servos can
actually do, and bake it into a **gait table** that the RL env imitates. It's a simplified take on
Disney's *Interactive Design of Stylized Walking Gaits for Robotic Characters* (SIGGRAPH 2024),
the pipeline behind their BDX droids.

```
style knobs ──► footsteps + LIPM/ZMP CoM plan + swing curves ──► whole-body IK ──► one gait cycle
     ▲                                                                              │
     └──────────── live 3D view, servo/balance checks, physics preview ◄────────────┘
                                         │ Bake
                                         ▼
          processed/gait_table.npz  ──►  BipedRobot env: imitation reward + RSI + gait clock
```

## Setup

```bash
cd BipedRobot/robot_motion_reference
python3 -m venv ~/.venvs/gait_studio
~/.venvs/gait_studio/bin/pip install -r gait_studio/requirements.txt   # Python 3.9: add "mujoco<3.3"
```

Keep the venv outside OneDrive.

## Run

```bash
cd BipedRobot/robot_motion_reference
~/.venvs/gait_studio/bin/python -m gait_studio.app        # open http://localhost:8080
```

### Drive tab
- **Command sliders and presets** set (vx, vy, wz). The cycle regenerates in about 30–50 ms.
- **Zero command means standing still**, as in training: the reference becomes the standing pose
  and the clock holds. Tick **Step in place when stopped** to preview the Base gait at zero command.
  Switching gaits blends the pose over 0.25 s instead of snapping.
- **Walk in place** keeps the robot centred and facing forward while the floor, footsteps and trails
  slide under it.
- **Reset drive** zeroes the command and restores the playback settings.
- **Auto tour** cycles through all the directions.
- **Pause and phase** let you scrub through a single cycle.
- **Overlays**:
  - footsteps (blue = left, orange = right)
  - swing-foot trails
  - CoM (green) and ZMP (red)
  - the support polygon
- **Physics preview** (orange robot) runs MuJoCo with the free-floating robot tracking the reference
  joints. It drives each joint through the training BAM servo model (`actuator.py`: firmware P, back-EMF,
  load-dependent friction), with no balance feedback. A gait that holds up open-loop for a few steps is a gait the policy can learn easily.
- **Clip ghost** (purple) plays an NPZ from `src/fbx_to_joint_data.py`, such as the game walk, next
  to the robot for style comparison. *Clip speed* compresses its timing: about 1.9× is Froude-correct
  going from a human to this robot.

### Style tab
Knobs come in three groups:

| Group | Knobs |
|---|---|
| Timing | cadence scale, speed → cadence, slowest / fastest cycle, max turn per step, double support |
| Feet | stance width, toe-out, step height, swing ease, apex timing, reach overshoot, bulge, side-step lift, heel-to-toe amount, ankle at contact, loading response, heel-off ankle angle, max leg reach, push-off |
| Body | CoM height, CoM forward, sway gain, ZMP inward, bob, trunk pitch, roll sway, twist, lean into motion |

The **Editing** dropdown picks what you edit:
- **Base** is the step-in-place gait.
- **Forward / Backward / Sideways / Turning** are anchors at the extremes of the command space.

A knob set on an anchor applies fully at that extreme and blends linearly back to the base value at
zero command.

**Resets:**
- A knob you change gets a "↺ reset" button under it. On Base it returns to the default; on an
  anchor it returns to that anchor's default.
- **Reset all knobs** restores the whole style, after a confirmation.

**Cadence** (Timing): the cycle length follows the command, as in humans, who split a speed change
between step length and step frequency (step length ∝ v^0.42–0.58: Grieve 1968, Kuo 2001, Wu et al.
2019). In leg-length units the law holds across body sizes (Hof 1996): `s/L = 1.15·v'^β` with
`v' = v/√(gL)`. This leg (L = 0.21 m) tops out at Fr ≈ 0.02, human very slow walking:

| command | cycle | step |
|---|---|---|
| vx 0.20 | 0.90 s | 9.0 cm |
| vx 0.10 | 1.27 s | 6.4 cm |
| vx 0.05 | 1.80 s | 4.5 cm |
| vy 0.15 | 0.70 s | sideways counts at 1/0.45 of its speed (Handford & Srinivasan 2014) |
| wz 1.2 in place | 0.73 s | quickened so one step turns at most *max turn per step* |

*Slowest cycle* caps the slow end (the human law gives ~1.9 s near the walk threshold), *fastest
cycle* the fast end, *speed → cadence* is 1−β (0 = constant cadence), *cadence scale* shifts the
whole curve. These knobs are global, the law already depends on the command. A style saved with the
old global `period` keeps exactly that cadence. The Drive tab's status line shows cycle, steps/min,
step length and Froude number.

**Heel-to-toe** comes from the rocker model (`rollover.py`), not from hand-set angles. Human stance
has three rockers (Perry): the leg arrives inclined with the ankle near neutral, so the foot lands
toes-up by the shank's lean and is lowered onto the heel (*loading response*); then the foot is flat
and the shank rolls over it; once the ankle has used its range (*heel-off ankle angle*, ~10° in
humans) the heel lifts and the foot pivots on its toe, plantarflexing further in pre-swing
(*push-off*). The ankle angle is foot pitch minus shank lean, so all of it follows from where the
leg is. Why humans do it: a rolling foot smooths the step-to-step transition and cuts the heel-strike
collision loss (Adamczyk, Collins & Kuo 2006), and it lengthens the effective leg. Robots add a
second trigger (Kajita; Miura et al. 2011): when the hip is further from the ankle than the leg
reaches (*max leg reach*), the foot pivots on its edge so the ankle rises toward the hip.
- The leg is evaluated from the plan (hip from the CoM target, ankle from the footstep, knee from the
  thigh/shank lengths), not from an IK solve, so a near-straight leg that cannot reach a step
  flat-footed does not feed its own errors back.
- Consequences, none of them authored: heel strike grows with step length (~10° at vx 0.1, ~19° at
  0.2; in humans the heel-strike angle is about the leg angle at touchdown, ~20°), the toes lead and the heel
  leaves last walking backward, and side steps and stepping in place stay flat (no sagittal shank lean
  in the foot's frame). *Ankle at contact* shifts the landing pitch, *heel-to-toe amount* scales it all
  (0 = flat feet).
- The pivot edge stays planted (the tests check this to within 1.5 mm), the swing starts and ends at the
  rolled poses, and the ZMP reference follows the contact (landing edge → sweep → leaving edge), so the CoM
  plan stays dynamically consistent. While a foot is pitched, its support shrinks to a 1 cm strip along
  the planted edge.
- It also helps the servos: at vx 0.1 the default style's torque-speed usage drops from ~310 % with flat
  feet to ~110 %.

**Direction-aware knobs.** Humans side-step with a nearly straight knee (knee range ~10° against ~60°
forward), getting clearance from hip abduction and a hip hike, and the leading foot lifts higher than
the closing one (Cappellini et al. 2023). So a sideways swing lifts *side-step lift* × step height
(the closing step 1/1.5 of that), the outward *bulge* fades out for sideways steps, and the Sideways
anchor gets a 2° *roll sway* (hip hike) by default. *Pelvis twist* is its value at max forward speed
scaled by the forward step length, since pelvic rotation follows sagittal step length (Huang et al.
2011): there is none in place or sideways. + = the swing-side hip leads at heel strike; at this
Froude number humans rotate the other way (negative).

**Why it sways, and how much.** Humans do sway: the whole-body CoM moves ~2.5–3.5 cm peak-to-peak
side to side at normal speed and up to ~10 cm at 0.3 m/s (Orendurff 2004, Staszkiewicz 2010, Tesio
& Rota 2019). It reads as "no sway" because it is ~2% of stature, smooth, and the head is
stabilised. It can't be avoided: on one foot the ZMP must be under that foot, so the inverted
pendulum accelerates the CoM toward it and back. In the LIPM the excursion is
`A = p·(1 − sech(ω·T_step/2))`, with `p` the ZMP's distance from the midline and `ω = √(g/z)`.
Two things made this robot sway several times more than a human for its leg length:
- **Feet too far apart.** The old 9 cm default stance is 0.43 leg lengths against the human 0.13, and
  wider than the 6.0 cm hip-pitch spacing, so the legs splayed. The new default is 6.0 cm.
- **ZMP at the sole centre.** Human feet nearly span their step width, so their pressure sits ~1 cm
  from the midline. *ZMP inward* (default 0.6) aims the stance ZMP toward the inner sole edge,
  keeping ~10 mm margin.

Together they halve the sway: at vx 0.1, 71 → 35 mm peak-to-peak, ~0.17 leg lengths. Lengthening
double support or reshaping the ZMP hand-over does not help (an exact periodic LIPM solve changes it
by 1–3 mm). *Sway gain* below 1 is not dynamically consistent: the ZMP leaves the foot, as the ZMP
margin check shows.

**Turning.** The body yaws over the planted stance foot, so seen from the body the foot slides front
to back. The centre of pressure rolls along the foot to stay under the body instead of dragging the
CoM along; before, that drag plus the sway made the body circle. Rocker pitches below ~1.5° are
treated as flat, so turning's tiny foot tilts don't throw the ZMP onto an edge.

"Lean into motion" is signed: set it on Sideways or Turning and it mirrors with direction. Selecting an
anchor also drives the robot to that command, so you see what you're editing.

### Check tab
Each check is graded 🟢 / 🟡 / 🔴 for the current command:

| Check | Notes |
|---|---|
| Joint limit margin | Against the *training* action limits |
| Peak joint speed | Against the lower of the BAM no-load speed (0.97·V/kt, 5.15 rad/s at 12 V) and the firmware slew limit (4.92 rad/s); warns at 2/3 |
| Peak torque | Inverse dynamics (with the BAM armature, 0.0234) plus the BAM friction the servo must overcome; warns at 70% of stall, fails at stall (2.94 N·m) |
| Torque-speed usage | \|τ_motor\| / (stall − sign(τ)·(kt²/R)·ω): back-EMF costs torque when driving and returns it when braking |
| ZMP margin | |
| Foot gap | |
| IK error | |

There are also plots for the selected joint.

### Bake tab
Save/load style presets (`styles/*.yaml`) and **Bake**. The same thing from the CLI:

```bash
python -m gait_studio.bake --style gait_studio/styles/default.yaml --out processed/gait_table.npz
```

Baking sweeps the grid in `configs/robot_v2.yaml` (9×7×13 commands, wz up to ±1.2 rad/s, about 13 s). Steps:
1. Generate and check one cycle per command.
2. Make the table and its clock frequency grid exactly left/right symmetric. The env's symmetry
   augmentation relies on this.
3. Write three files:
   - `gait_table.npz`
   - `gait_table_style.yaml` (the exact style that was baked)
   - `gait_table_report.txt` (a feasibility map)

`*.npz` is gitignored. Commit the style yaml, or force-add the table.

## What's in the table

`data[n_vx, n_vy, n_wz, n_phase, 44]` (table v2): every cycle is stored **phase-normalised**,
`n_phase` = 64 samples per cycle, because each command has its own period. `freq[n_vx, n_vy, n_wz]`
is the gait clock frequency (cycles/s) per command. Frame layout (`gait.FIELDS`):

| field | dims | notes |
|---|---|---|
| joint_pos, joint_vel | 12, 12 | Isaac joint order, bound by name |
| base_height | 1 | base_link origin above the ground (used for RSI) |
| projected_gravity | 3 | base frame |
| base_lin_vel_b, base_ang_vel_b | 3, 3 | base frame |
| foot_pos_b | 6 | foot body origins in the base frame, [left, right] |
| foot_height | 2 | sole above the ground |
| contact | 2 | 1 = stance |

Plus `stand` (the standing frame), the grid axes, `period` (the zero-command cycle, display only), the
field layout, and the per-command feasibility status and metrics.

**Phase convention:**
- Phase 0 is a right-foot touchdown.
- [0, 0.5) is right stance and [0.5, 1) is left stance.
- In the env's terms: `sin(phi) > 0` means right stance, and a mirror is phi + π.

## How training uses it

See `BipedRobot/config/config_gait_reference.yaml`.

- **Gait clock.** It advances by `2π·dt·freq(cmd)`, with `freq` interpolated trilinearly in the
  command and clamped to the grid (the same rule as the table), and holds while the command is below
  `stand_threshold`. A v1 table runs at one period. The policy still sees `[sin, cos]`, so the obs
  size doesn't change.
- **Imitation rewards.** The `imitation_*` terms are exp kernels: joint pos/vel, foot positions in
  the base frame, orientation, base velocity, and contact match. They compare against
  `reference(command, phase)`, interpolated on the GPU (trilinear in the command, linear in phase).
- **RSI.** 80% of resets start mid-stride from the reference pose.
- **Pruned terms.** The hand-written gait terms are zeroed: `swing_foot_height`,
  `gait_phase_contact`, `feet_air_time`, the `joint_deviation_*` terms, and `foot_separation`.

Nothing from the table is needed on the robot. The runtime only needs the same clock rule, which
`export_onnx.py` writes into `deploy_config.json` under `gait_clock`: `cadence_grid` (axes + freq,
bundle format v5) for a v2 table whose cadence varies, `fixed_period` otherwise.

## What the analyzer says about this robot

With the BAM STS3215 model (stall capped at 2.94 N·m, armature 0.0234 kg·m²), the swing leg is the bottleneck:
- The shank is short, so a 1 cm foot lift already takes about 20° of extra knee bend.
- The average swing-foot speed is 2·v/(1−double_support) whatever the period.
- Reflected rotor inertia makes every rad/s² cost about 0.023 N·m, and BAM friction adds roughly 0.15 N·m
  plus a load-dependent share on top of every moving joint.

The numbers in this section were measured with the older 0.04 armature / 4.71 rad/s model; re-bake to
see where the BAM model puts the limits.

Clean references exist up to roughly |vx| 0.06–0.1 m/s, |vy| 0.05 m/s and |wz| 1.0 rad/s. The default
grid stretches a bit past that and the bake report flags those cells. Things that help:
- narrower stance (lower hip-roll load)
- lower step height
- lower swing ease (lower peak speed)
- a smaller armature

The servo comes from `actuator:` in `configs/robot_v2.yaml`, which points at the same BAM params file
the training env uses (`actuator/params/sts3215_12v/m6.json`, `kp_fw`, `stall_torque_cap_nm`). Keep it
in sync with `config/config.yaml` (`env.actuator`); `actuator.vin` is the nominal supply (training
samples 11.1–12.6 V, so margins near the limit are tighter at the low end).

## Files

| file | what |
|---|---|
| `robot.py` | Loads the training MJCF, re-points mesh paths, measures the soles, adds sole sites |
| `style.py` | Knob definitions, anchors, blending, yaml I/O |
| `cadence.py` | Speed-dependent cycle length (human gait scaling) |
| `footsteps.py` | Path integration, footstep schedule, swing curves, ZMP reference |
| `rollover.py` | Rocker model: stance-foot pitch (heel strike, heel off) from the leg geometry |
| `lipm.py` | ZMP preview control (Kajita 2003) |
| `ik.py` | Floating-base damped least-squares IK with an active set on joint limits |
| `gait.py` | `GaitGenerator.cycle()` / `.stand()` and the table frame layout |
| `analysis.py` | Feasibility checks, including inverse-dynamics torques with ZMP-split contact wrenches |
| `actuator.py` | Numpy port of the training BAM servo (`actuator/bam_kernel.py`): torque-speed envelope, friction budget, firmware loop |
| `physics.py` | Open-loop MuJoCo preview |
| `bake.py` | Grid bake, symmetrization, report |
| `app.py` | The viser web UI |
| `tests/` | Symmetry, periodicity, velocity tracking, phase convention, table symmetry, cadence, rocker model, side steps |

Run the tests from `robot_motion_reference/`:

```bash
python -m pytest gait_studio/tests -q
```
