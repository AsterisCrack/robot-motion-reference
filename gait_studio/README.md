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
  joints. It uses the training PD (kp 20, kd 2) and the servo torque-speed envelope, with no balance
  feedback. A gait that holds up open-loop for a few steps is a gait the policy can learn easily.
- **Clip ghost** (purple) plays an NPZ from `src/fbx_to_joint_data.py`, such as the game walk, next
  to the robot for style comparison. *Clip speed* compresses its timing: about 1.9× is Froude-correct
  going from a human to this robot.

### Style tab
Knobs come in three groups:

| Group | Knobs |
|---|---|
| Timing | period, double support |
| Feet | stance width, toe-out, step height, swing ease, apex timing, reach overshoot, bulge, heel strike, toe off, heel/toe roll time |
| Body | CoM height, CoM forward, sway gain, bob, trunk pitch, roll sway, twist, lean into motion |

The **Editing** dropdown picks what you edit:
- **Base** is the step-in-place gait.
- **Forward / Backward / Sideways / Turning** are anchors at the extremes of the command space.

A knob set on an anchor applies fully at that extreme and blends linearly back to the base value at
zero command.

**Resets:**
- A knob you change gets a "↺ reset" button under it. On Base it returns to the default; on an
  anchor it returns to that anchor's default.
- **Reset all knobs** restores the whole style, after a confirmation.

**Heel-to-toe** (heel strike, toe off, roll times):
- The stance foot lands toes-up and rolls down about its heel edge, then peels the heel up and
  leaves off the toe. Walking backward reverses this.
- The pivot edge stays planted (the tests check this to within 1.5 mm).
- The swing starts and ends at the rolled poses.
- The ZMP reference follows the contact region (heel edge → sweep → toe edge), so the CoM plan stays
  dynamically consistent. While a foot is rolled, the support polygon shrinks to a 1 cm strip along
  the planted edge.
- By default it is set on the Forward/Backward anchors only, so stepping in place stays flat.
- It also helps the servos: toe-off lengthens the effective leg. At 0.2 m/s, peak joint speed
  drops from about 6 to 3.5 rad/s. "Lean into motion" is signed: set it on Sideways or Turning and it mirrors with
direction. Selecting an anchor also drives the robot to that command, so you see what you're editing.

### Check tab
Each check is graded 🟢 / 🟡 / 🔴 for the current command:

| Check | Notes |
|---|---|
| Joint limit margin | Against the *training* action limits |
| Peak joint speed | Against 4.71 rad/s no-load |
| Peak torque | Estimated by inverse dynamics, including the 0.04 armature |
| Torque-speed usage | \|τ\| / (stall·(1−\|ω\|/ω_max)) |
| ZMP margin | |
| Foot gap | |
| IK error | |

There are also plots for the selected joint.

### Bake tab
Save/load style presets (`styles/*.yaml`) and **Bake**. The same thing from the CLI:

```bash
python -m gait_studio.bake --style gait_studio/styles/default.yaml --out processed/gait_table.npz
```

Baking sweeps the grid in `configs/robot_v2.yaml` (11×7×13 commands, wz up to ±1.2 rad/s, about 6 s). Steps:
1. Generate and check one cycle per command.
2. Make the table exactly left/right symmetric. The env's symmetry augmentation relies on this.
3. Write three files:
   - `gait_table.npz`
   - `gait_table_style.yaml` (the exact style that was baked)
   - `gait_table_report.txt` (a feasibility map)

`*.npz` is gitignored. Commit the style yaml, or force-add the table.

## What's in the table

`data[n_vx, n_vy, n_wz, n_frames, 44]`, at 50 Hz over one period. Frame layout (`gait.FIELDS`):

| field | dims | notes |
|---|---|---|
| joint_pos, joint_vel | 12, 12 | Isaac joint order, bound by name |
| base_height | 1 | base_link origin above the ground (used for RSI) |
| projected_gravity | 3 | base frame |
| base_lin_vel_b, base_ang_vel_b | 3, 3 | base frame |
| foot_pos_b | 6 | foot body origins in the base frame, [left, right] |
| foot_height | 2 | sole above the ground |
| contact | 2 | 1 = stance |

Plus `stand` (the standing frame), the grid axes, `period`, the field layout, and the per-command
feasibility status and metrics.

**Phase convention:**
- Phase 0 is a right-foot touchdown.
- [0, 0.5) is right stance and [0.5, 1) is left stance.
- In the env's terms: `sin(phi) > 0` means right stance, and a mirror is phi + π.

## How training uses it

See `BipedRobot/config/config_gait_reference.yaml`.

- **Gait clock.** It runs at 1/period and holds while the command is below `stand_threshold`. This
  replaces the speed-proportional clock. The policy still sees `[sin, cos]`, so the obs size doesn't
  change.
- **Imitation rewards.** The `imitation_*` terms are exp kernels: joint pos/vel, foot positions in
  the base frame, orientation, base velocity, and contact match. They compare against
  `reference(command, phase)`, interpolated on the GPU (trilinear in the command, linear in phase).
- **RSI.** 80% of resets start mid-stride from the reference pose.
- **Pruned terms.** The hand-written gait terms are zeroed: `swing_foot_height`,
  `gait_phase_contact`, `feet_air_time`, the `joint_deviation_*` terms, and `foot_separation`.

Nothing from the table is needed on the robot. The runtime only needs the same clock rule, which
`export_onnx.py` writes into `deploy_config.json` under `gait_clock`.

## What the analyzer says about this robot

With STS3215 limits and the sim's armature of 0.04 kg·m², the swing leg is the bottleneck:
- The shank is short, so a 1 cm foot lift already takes about 20° of extra knee bend.
- The average swing-foot speed is 2·v/(1−double_support) whatever the period.
- Reflected rotor inertia makes every rad/s² cost about 0.04 N·m.

Clean references exist up to roughly |vx| 0.06–0.1 m/s, |vy| 0.05 m/s and |wz| 1.0 rad/s. The default
grid stretches a bit past that and the bake report flags those cells. Things that help:
- narrower stance (lower hip-roll load)
- lower step height
- lower swing ease (lower peak speed)
- a smaller armature

BAM identifies the STS3215 (7.4 V) at about 0.024 kg·m². Set `actuator.armature` in
`configs/robot_v2.yaml` to whatever your new BAM-based sim uses, so the checks match your training
physics.

## Files

| file | what |
|---|---|
| `robot.py` | Loads the training MJCF, re-points mesh paths, measures the soles, adds sole sites |
| `style.py` | Knob definitions, anchors, blending, yaml I/O |
| `footsteps.py` | Path integration, footstep schedule, swing curves, ZMP reference |
| `lipm.py` | ZMP preview control (Kajita 2003) |
| `ik.py` | Floating-base damped least-squares IK with an active set on joint limits |
| `gait.py` | `GaitGenerator.cycle()` / `.stand()` and the table frame layout |
| `analysis.py` | Feasibility checks, including inverse-dynamics torques with ZMP-split contact wrenches |
| `physics.py` | Open-loop MuJoCo preview |
| `bake.py` | Grid bake, symmetrization, report |
| `app.py` | The viser web UI |
| `tests/` | Symmetry, periodicity, velocity tracking, phase convention, table symmetry |

Run the tests from `robot_motion_reference/`:

```bash
python -m pytest gait_studio/tests -q
```
