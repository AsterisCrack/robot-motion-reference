# Gait Studio

**Design a walking style for a biped in the browser, check it against what its servos can do, and bake it into a gait table that a reinforcement-learning environment can imitate.**

<p align="center">
  <img src="https://raw.githubusercontent.com/AsterisCrack/BipedRobot/main/media/gait_studio.gif" alt="Gait Studio: interactive gait design in the browser" width="100%"/>
</p>

Gait Studio is a simplified take on Disney Research's *Interactive Design of Stylized Walking Gaits for
Robotic Characters* (SIGGRAPH 2024). You set a handful of style knobs. It plans footsteps, a balanced
centre-of-mass motion and whole-body inverse kinematics for any velocity command, and shows the result live.

```
style knobs ──► footsteps + LIPM/ZMP CoM plan + swing curves ──► whole-body IK ──► one gait cycle
     ▲                                                                              │
     └────────── live 3D view, servo and balance checks, physics preview ◄──────────┘
                                        │ Bake
                                        ▼
        gait_table.npz ──► training env: imitation reward + reset-from-reference + gait clock
```

- Runs on a laptop: Python and MuJoCo only, no Isaac Lab and no GPU.
- A gait is generated from the command in a fraction of a second, so the sliders feel live.
- The baked table holds one cycle per (vx, vy, yaw-rate) command, exactly left/right symmetric.

---

## Setup

```bash
cd robot_motion_reference            # the repo root (BipedRobot/robot_motion_reference as a submodule)
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r gait_studio/requirements.txt              # on Python 3.9, also pin "mujoco<3.3"
```

Dependencies: `mujoco`, `viser`, `numpy`, `scipy`, `pyyaml`, `pytest`.

The bundled robot config describes the [BipedRobot](https://github.com/AsterisCrack/BipedRobot) V2. It reads the
robot's MJCF and servo parameters from that repo, so clone it with its submodules and work from there:

```bash
git clone --recurse-submodules https://github.com/AsterisCrack/BipedRobot
cd BipedRobot/robot_motion_reference
```

For any other robot, see [Robot config](#robot-config).

## Run the app

```bash
python -m gait_studio.app            # then open http://localhost:8080
```

| Flag | Default | |
|---|---|---|
| `--robot` | `configs/robot_v2.yaml` | [Robot config](#robot-config) |
| `--style` | `styles/default.yaml` | Style preset to start from |
| `--port` | `8080` | |
| `--host` | `127.0.0.1` | Use `0.0.0.0` to open the Studio from another machine |

The UI has four tabs.

### Drive

Pick a command and watch the gait.

| Control | What it does |
|---|---|
| Forward, lateral and turn sliders | Set the command (vx, vy, wz) within the robot config's `command_ranges` |
| Preset buttons | Stop, forward, back, left, right, turn left, turn right |
| Effective command | Shown when a command limit is active: the slider command after it is scaled into the limit (see [Command limit](#command-limit)) |
| Auto tour | Cycles through all the presets |
| Walk in place | Keeps the robot centred and facing forward while the floor, footsteps and trails slide under it |
| Step in place when stopped | At zero command the robot stands still, as in training. Tick this to preview the base stepping gait instead. |
| Playback speed, Pause, Phase | Slow down or scrub through one cycle |
| Footsteps, swing trails, CoM / ZMP / support | Overlays: footsteps (blue left, orange right), swing-foot trails, CoM (green), ZMP (red), support polygon |
| Physics preview | An orange robot in MuJoCo tracks the reference joints through the servo model, with no balance feedback. A gait that survives a few steps open-loop is one a policy can learn easily. |
| Clip ghost | Plays a joint-angle `.npz` (see the [FBX pipeline](../README.md#fbx-retargeting-pipeline)) as a purple robot next to the generated gait, for style comparison. Type the file path in the field (relative paths start at the repo root; the field is pre-filled with `processed/StandardWalk.npz`, which is not shipped). Joints are matched by name against the robot config's `joints`. *Clip speed* compresses its timing. About 1.9x is the Froude-correct factor from a human to the bundled robot. |

Changing a gait blends the pose over a quarter of a second instead of snapping. **Reset drive** zeroes the
command and the playback settings.

### Style

Every knob that shapes the gait. Hover a slider in the app for its description.

**Timing** (global: the cadence law already depends on the command)

| Knob | Default | Effect |
|---|---|---|
| Cadence scale | 1.0 | Multiplies the human step frequency. Above 1 means quicker, shorter steps. |
| Speed to cadence | 0.5 | Cadence grows as speed to this power. 0 is a constant cadence. |
| Slowest cycle / Fastest cycle | 2.0 s / 0.5 s | Clamp the cycle length at the slow and fast end |
| Max turn per step | 25 deg | Quickens the cadence when turning so one step never turns more than this |
| Double support | 0.10 | Fraction of each step with both feet down |

**Feet**

| Knob | Default | Effect |
|---|---|---|
| Stance width | 0.06 m | Lateral distance between sole centres |
| Min foot gap | 0.012 m | Clearance kept between the inner sole edges (global) |
| Toe-out | 0 deg | Foot yaw outward |
| Step height | 0.01 m | Swing-foot apex above the ground |
| Swing ease | 0.20 | Share of the swing spent speeding up and slowing down. Lower means lower peak joint speed but a harsher start and stop. |
| Apex timing | 0.5 | When in the swing the apex happens |
| Reach overshoot | 0 | Foot reaches past its target and settles back |
| Swing bulge | 0 m | Outward arc of the swing foot |
| Side-step lift | 0.6 | Foot lift on sideways steps, relative to step height |
| Heel-to-toe amount | 1.0 | Scales the heel strike and toe-off the rocker model derives. 0 gives flat feet. |
| Ankle at contact | 0 deg | Ankle angle at touchdown, relative to mid-stance |
| Loading response | 0.17 | Share of stance spent lowering the foot from heel contact to flat |
| Heel-off ankle angle | 10 deg | How far the shank may lean over the flat foot before the heel lifts |
| Max leg reach | 0.97 | Fraction of the straight leg beyond which the foot pivots on its edge (global) |
| Push-off | 0 deg | Extra plantarflexion through the last double support |

**Body**

| Knob | Default | Effect |
|---|---|---|
| CoM height | 0.195 m | Lower means more knee bend |
| CoM forward offset | 0 m | Shifts the CoM ahead of the feet |
| Lateral sway gain | 1.0 | Scales the side-to-side CoM motion. 1 is dynamically consistent. |
| ZMP inward | 0.6 | Where the ZMP sits across the stance foot: 0 is the sole centre, 1 is the inner edge |
| Vertical bob, Bob phase | 0 m, 0 deg | CoM up and down, twice per cycle |
| Trunk pitch | 0 deg | Forward torso lean |
| Roll sway | 0 deg | Torso roll toward the stance foot |
| Pelvis twist | 0 deg | Pelvis yaw, scaled by the forward step length |
| Lean into motion | 0 deg | Torso roll into sidesteps and turns. Set on an anchor. |

The **Editing** dropdown picks what the sliders edit:

- **Base** is the step-in-place gait (zero command).
- **Forward, Backward, Sideways, Turning** are *anchors* at the extremes of the command space. A knob set on
  an anchor applies fully at that extreme and blends linearly back to the base value toward zero command.
  Selecting an anchor also drives the robot to that command (untick *Drive to the anchor command* to stay put).
  Signed knobs mirror with direction: a lean set on Sideways leans left when stepping left and right when
  stepping right.

A knob that differs from its default shows a reset button under it. **Clear overrides on this anchor** and
**Reset all knobs** restore the defaults.

### Check

Every check is graded green, yellow or red for the current command.

| Check | Yellow | Red |
|---|---|---|
| Joint limit margin (against the training action limits) | under 2 deg | past the limit |
| Peak joint speed | above 2/3 of the servo limit | above the limit (the lower of the no-load speed and the firmware slew rate) |
| Peak joint torque (inverse dynamics plus the friction the servo must overcome) | above 70% of stall | above stall |
| Torque-speed usage (torque against what the servo can deliver at that speed) | above 70% | above 100% |
| ZMP margin (inside the support polygon) | under 5 mm | under -5 mm |
| Min foot gap | under 5 mm | overlapping |
| IK foot and CoM error | above 2 mm | above 5 mm |
| IK stance tilt error | above 1 deg | above 3 deg |

Plots for a selected joint show its angle against the limits, its speed against the servo limit, its torque
usage, and the foot heights with the ZMP margin.

### Bake

Save and load style presets, then bake. Details in [Baking a gait table](#baking-a-gait-table).

---

## Styles

A style is a YAML file: the base knobs plus per-anchor overrides. Presets live in
[`styles/`](styles/) and the **Bake** tab saves new ones there.

```yaml
name: default
base:
  stance_width: 0.06
  step_height: 0.01
  sway_gain: 1.0
  # ... every knob not listed keeps its default
anchors:
  sideways:
    trunk_roll_sway: 2.0       # a 2 deg hip hike at full sideways speed, fading to 0 at standing
```

| Preset | Character |
|---|---|
| [`default`](styles/default.yaml) | Narrow stance under the hips, low steps, human cadence law |
| [`natural`](styles/natural.yaml) | Wider stance, 0.6 to 0.9 s cycles, push-off |
| [`stable`](styles/stable.yaml) | Constant cadence, long double support, higher steps |

Unknown knobs are ignored. Styles written for older versions load too: a fixed `period` becomes a cadence
pinned to that period, and retired heel-to-toe angles are dropped with a notice.

---

## How the gait is generated

A gait comes from a pipeline of small, separately tested pieces ([file map](#file-map)). The design choices
that make it look human, and keep it within the reach of small servos:

- **Cadence follows speed, as in humans** ([`cadence.py`](cadence.py)). Humans split a speed change between
  step length and step frequency: step length grows as speed to the power of about 0.42 to 0.58 (Grieve 1968;
  Kuo 2001; Wu et al. 2019). In leg-length units the law holds across body sizes (Hof 1996):
  `s/L = 1.15 * v'^beta`, with `v' = v / sqrt(g L)`. So a slow command gets shorter steps at a lower
  cadence, not full-length steps in slow motion. Sideways speed counts at 1/0.45 of its value (Handford & Srinivasan 2014), and turning
  keeps cadence roughly constant (Courtine & Schieppati 2003). The *cadence* knobs shift or clamp the curve.
- **Heel-to-toe from a rocker model** ([`rollover.py`](rollover.py)). Human stance has three rockers (Perry):
  the foot lands toes-up by the shank's lean and is lowered onto the heel, the shank then rolls over the flat
  foot, and once the ankle has used its range the heel lifts and the foot pivots on its toe. The ankle angle is
  the foot pitch minus the shank lean, so all of it follows from the leg geometry instead of hand-set angles. A
  rolling foot smooths the step-to-step transition and cuts the heel-strike collision loss (Adamczyk, Collins
  & Kuo 2006). Robots add a second trigger (Kajita; Miura et al. 2011): when the hip is further from the ankle
  than the leg reaches, the foot pivots on its edge so the ankle rises toward the hip. Heel strike therefore
  grows with step length, reverses when walking backward, and vanishes in place and on side steps. The pivot
  edge stays planted and the ZMP reference follows the contact edge, so the CoM plan stays consistent.
- **Direction-aware steps.** Humans side-step with a nearly straight knee, getting clearance from hip
  abduction and a hip hike, and the leading foot lifts higher than the closing one (Cappellini et al. 2023).
  So sideways swings lift by *side-step lift* times the step height, the outward bulge fades out, and the
  Sideways anchor adds a small roll sway. Pelvis twist scales with the forward step length (Huang et al. 2011),
  so there is none in place or sideways.
- **LIPM/ZMP sway** ([`lipm.py`](lipm.py)). On one foot the ZMP has to be under that foot, so a linear
  inverted pendulum accelerates the CoM toward it and back. Humans sway a few centimetres side to side
  (Orendurff 2004; Staszkiewicz 2010; Tesio & Rota 2019), and the sway is unavoidable. Its size is set by
  where the ZMP sits, so a narrow stance under the hips and a ZMP aimed toward the inner sole edge
  (*ZMP inward*) keep it human-proportioned. The CoM follows the ZMP reference through preview control
  (Kajita et al. 2003). A *sway gain* below 1 makes the ZMP leave the foot, which the ZMP margin check flags.
- **Turning.** The body yaws over the planted foot and the centre of pressure rolls along the foot to stay
  under the body, so the body does not circle.
- **Servo-aware by construction.** Swing curves have an eased trapezoidal speed profile to cap peak joint
  speed, and whole-body IK solves both soles, the CoM and the torso together with joint limits as an active
  set. The checks above then price the result in the servo's torque and speed.

---

## Baking a gait table

From the **Bake** tab, or the CLI:

```bash
python -m gait_studio.bake --style gait_studio/styles/default.yaml --out processed/gait_table.npz
```

| Flag | Default | |
|---|---|---|
| `--style` | `styles/default.yaml` | Style to bake |
| `--robot` | `configs/robot_v2.yaml` | Robot config (sets the command grid and limits) |
| `--out` | `processed/gait_table.npz` | Output table |
| `--workers` | CPU count minus one | Parallel processes |
| `--grid NVX NVY NWZ` | from the robot config | Points per axis, ends included |
| `--limit-shape {box,ellipse_xy}` | from the robot config | [Command limit](#command-limit) shape for this bake |

The bake generates and checks one cycle for every grid command in parallel, makes the table and its clock
frequencies exactly left/right symmetric (mirror augmentation in training relies on this), and writes three
files next to each other:

| File | |
|---|---|
| `gait_table.npz` | The table |
| `gait_table_style.yaml` | The exact style that was baked |
| `gait_table_report.txt` | Feasibility report: ok/warn/fail counts, a status map, cycle periods and torque-speed usage across the command grid |

Cells marked warn or fail in the report are commands the servo cannot fully deliver. Commit the style yaml:
`*.npz` is gitignored.

### Command limit

`command_limit_shape` in the robot config is `ellipse_xy` or `box`. With `ellipse_xy`, commands outside
`(vx/vx_max)^2 + (vy/vy_max)^2 <= 1` are not asked for in training, so those grid cells are baked at their
command scaled onto the ellipse. The blend near the boundary stays feasible and no impossible corner gait
is generated. The report counts these cells. [`command_limit.py`](command_limit.py) is the NumPy implementation.

This is only a mirror of the training setting. The training config stays the source of truth and should use
the same shape and ranges, and the table records both for a consistency check.

---

## The table format

`data[n_vx, n_vy, n_wz, n_phase, 44]`, float32. Every cycle is stored **phase-normalised** with `n_phase`
samples (64 by default), because every command has its own period. `freq[n_vx, n_vy, n_wz]` is the gait clock
frequency in cycles per second.

One 44-value frame (`gait.FIELDS`):

| Field | Size | Notes |
|---|---|---|
| `joint_pos`, `joint_vel` | 12, 12 | In the table's `joint_names` order; bind by name |
| `base_height` | 1 | Base origin above the ground (for reset from reference) |
| `projected_gravity` | 3 | Base frame |
| `base_lin_vel_b`, `base_ang_vel_b` | 3, 3 | Base frame |
| `foot_pos_b` | 6 | Foot body origins in the base frame, [left, right] |
| `foot_height` | 2 | Sole above the ground, [left, right] |
| `contact` | 2 | 1 means stance, [left, right] |

The file also holds:

| Key | |
|---|---|
| `version` | `3` (readers should also accept 1 and 2) |
| `stand` | The standing frame, used at zero command |
| `grid_vx`, `grid_vy`, `grid_wz` | Grid axes |
| `field_names`, `field_widths`, `joint_names` | Frame layout |
| `period`, `control_dt`, `n_frames` | Zero-command cycle length (display only), control step, samples per cycle |
| `phase_convention` | Text description of the convention below |
| `command_limit_shape`, `command_ranges_x/y/z`, `cmd_baked` | Command-limit metadata; `cmd_baked` is the command each cell was baked at |
| `status`, `metric_*` | Per-command feasibility status and check values |
| `style_yaml`, `style_digest` | The style that produced the table |

Versions: 1 is a single fixed period, 2 adds the per-command `freq`, 3 adds the command-limit keys.

**Phase convention.** Phase 0 is a right-foot touchdown. The interval [0, 0.5) is right stance and [0.5, 1)
is left stance. The mirror of a gait is the same cycle at phase + 0.5 with left and right swapped and
(vy, wz) negated.

Reading a cycle without any framework:

```python
import numpy as np

t = np.load("processed/gait_table.npz")
nearest = lambda axis, v: np.abs(t[axis] - v).argmin()
i, j, k = nearest("grid_vx", 0.1), nearest("grid_vy", 0.0), nearest("grid_wz", 0.0)

cycle = t["data"][i, j, k]                              # (n_phase, 44)
period = 1.0 / t["freq"][i, j, k]                       # seconds
ends = np.cumsum(t["field_widths"])
joint_pos = cycle[:, : ends[0]]                         # (n_phase, 12)
```

## Using a table in training

Nothing from the table is needed on the robot. A training environment needs four things:

- **A gait clock.** Advance the phase by `dt * freq(command)` each step, with `freq` interpolated
  trilinearly in the command and clamped to the grid. Hold the clock, and use `stand`, while the command is
  below the config's `stand_threshold`. The policy sees `[sin(2 pi phase), cos(2 pi phase)]`.
- **A reference lookup.** `reference(command, phase)`: trilinear in the command, linear in phase (wrapping).
- **Imitation rewards.** Exponential kernels comparing the robot to the reference: joint positions and
  velocities, foot positions in the base frame, orientation, base velocity and contact.
- **Reset from reference.** Start a share of episodes mid-stride from the reference pose, using
  `joint_pos`, `joint_vel` and `base_height`.

At deployment only the clock rule is needed. The BipedRobot implementation is
[`utils/motion_reference.py`](https://github.com/AsterisCrack/BipedRobot/blob/main/utils/motion_reference.py),
configured as in
[`config/config_gait_reference.yaml`](https://github.com/AsterisCrack/BipedRobot/blob/main/config/config_gait_reference.yaml).

---

## Robot config

Everything robot-specific is in one YAML file, [`configs/robot_v2.yaml`](configs/robot_v2.yaml). Paths in it
are relative to the file.

| Key | |
|---|---|
| `mjcf`, `mesh_dir` | The robot's MuJoCo model and its meshes |
| `base_body`, `foot_bodies` | Base body name and `{left, right}` foot body names |
| `joints` | The 12 joints, in table order |
| `joint_limits_deg` | The training action limits per joint. A reference outside them cannot be reached by the policy. |
| `actuator` | Servo model: `params_file`, `kp_fw`, `stall_torque_cap_nm`, `vin`, optional `armature` |
| `command_ranges` | `vx`, `vy`, `wz` ranges covered by the table. Also where the style anchors sit. |
| `command_limit_shape` | `ellipse_xy` or `box` |
| `grid` | Points per axis for a bake |
| `n_phase` | Samples per cycle (even) |
| `stand_threshold` | Command magnitude below which the robot stands |
| `control_dt` | Policy step |

The sole rectangle is not configurable: it is always measured from the foot meshes. (`robot_v2.yaml` carries a
`sole: auto: true` entry, which the code does not read.)

The bundled config points at the BipedRobot repo (`../../../envs/assets/robotV2/` and
`../../../actuator/params/`). Keep the servo settings and command ranges in sync with the training config.

### Adapting it to another robot

Copy `robot_v2.yaml`, point it at your model, and run the app with `--robot`. The generator is written for
this kind of robot, so check your model against these requirements:

- **Twelve actuated joints**, six per leg, named with an `l_` or `r_` prefix. Each leg needs exactly one joint
  whose name contains `hip_pitch`, one `knee` and one `ankle_pitch`. The 12-joint count is built into the
  table layout.
- **Mirrored legs.** Left and right joint values must mirror by a plain swap, as in a mirrored CAD. Joints that
  need a sign flip are not supported without editing `robot.py`.
- **A free-floating base** as the first joint of the MJCF, with the robot facing +x, left on +y, z up.
- **Foot bodies with mesh geoms.** The sole rectangle is measured from the lowest 3 mm of the foot meshes.
  Mesh paths in the MJCF that end in `.../meshes/` are redirected to `mesh_dir`.
- **MJCF actuators named after their joints**, for the physics preview.
- **Knees that bend toward negative angles.** The IK starts from a bent-knee seed with a negative knee angle.
- **A servo parameter file** in the BAM format (Rhoban's [BAM](https://github.com/Rhoban/bam)): a JSON with
  `model` (`m1` to `m6`), `kt`, `R`, `error_gain_ratio`, `armature`, `max_velocity`, `command_delay`,
  `friction_base`, `friction_viscous`, plus the friction terms the chosen model needs (see
  [`actuator.py`](actuator.py)). One servo model is shared by all joints. The BipedRobot repo has tools to
  identify one from bench recordings.

Check the `command_ranges` against what the checks report as feasible for your servos, and widen them only as
far as the Check tab and the bake report stay green.

---

## File map

| File | What |
|---|---|
| [`app.py`](app.py) | The browser UI ([viser](https://viser.studio)) |
| [`bake.py`](bake.py) | Grid bake, projection to the command limit, symmetrisation, report |
| [`style.py`](style.py) | Knob definitions, anchors, blending, YAML I/O |
| [`gait.py`](gait.py) | `GaitGenerator.cycle()` and `.stand()`, and the table frame layout |
| [`cadence.py`](cadence.py) | Speed-dependent cycle length |
| [`footsteps.py`](footsteps.py) | Path integration, footstep schedule, swing curves, ZMP reference |
| [`rollover.py`](rollover.py) | Rocker model: stance-foot pitch from the leg geometry |
| [`lipm.py`](lipm.py) | ZMP preview control of the CoM |
| [`ik.py`](ik.py) | Floating-base damped least-squares IK with joint limits as an active set |
| [`analysis.py`](analysis.py) | The feasibility checks, including inverse-dynamics torques |
| [`actuator.py`](actuator.py) | NumPy port of the BAM servo model |
| [`physics.py`](physics.py) | Open-loop MuJoCo preview |
| [`robot.py`](robot.py) | Loads the MJCF, measures the soles, joint bookkeeping |
| [`command_limit.py`](command_limit.py) | Command limit (`box`, `ellipse_xy`) |
| [`configs/`](configs/), [`styles/`](styles/) | Robot config, style presets |
| [`tests/`](tests/) | Symmetry, periodicity, velocity tracking, phase convention, table symmetry, cadence, rocker model, side steps, command limit, servo model |

Run the tests from the repo root:

```bash
python -m pytest gait_studio/tests -q
```
