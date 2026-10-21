"""Generate one steady-state gait cycle (or the standing pose) for a command and style."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .cadence import period_for
from .footsteps import GaitPlan, convex_hull, rotz
from .ik import IKTargets, WholeBodyIK, bent_knee_pose
from .lipm import G, preview_com
from .robot import SIDES, RobotModel
from .rollover import Rollover
from .style import Style

# Layout of one reference frame. The env reads these by name from the baked npz, so the
# order here is free to change as long as FIELDS stays the single source of truth.
FIELDS: list[tuple[str, int]] = [
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("base_height", 1),       # base_link origin above the ground plane
    ("projected_gravity", 3), # gravity unit vector in the base frame
    ("base_lin_vel_b", 3),
    ("base_ang_vel_b", 3),
    ("foot_pos_b", 6),        # foot BODY origins (as the env reads body_pos_w) in the base frame, [left xyz, right xyz]
    ("foot_height", 2),       # sole height above the ground, [left, right]
    ("contact", 2),           # 1 = stance, [left, right]
]
FIELD_SLICES: dict[str, slice] = {}
_o = 0
for _n, _w in FIELDS:
    FIELD_SLICES[_n] = slice(_o, _o + _w)
    _o += _w
FRAME_DIM = _o

PLANNER_DT = 0.005


def sample_com(com: np.ndarray, t_start: float, t: float) -> np.ndarray:
    """CoM xy of a preview plan (samples [pos, vel, acc] every PLANNER_DT from t_start) at time t.

    Frames sit at period/n_frames, which is generally not a multiple of PLANNER_DT. Rounding to
    the nearest sample jitters the position by up to half a sample, and the frames' second
    difference turns that into a ZMP error of centimetres. The cart-table model's jerk is
    piecewise constant, so the exact in-between value is a cubic.
    """
    x = (t - t_start) / PLANNER_DT
    k = int(min(max(math.floor(x), 0), com.shape[0] - 2))
    tau = (x - k) * PLANNER_DT
    pos, vel, acc = com[k, :, 0], com[k, :, 1], com[k, :, 2]
    jerk = (com[k + 1, :, 2] - acc) / PLANNER_DT
    return pos + vel * tau + acc * tau * tau / 2.0 + jerk * tau ** 3 / 6.0

def mirror_rows(frames: np.ndarray, robot: RobotModel) -> np.ndarray:
    """Left/right reflection of table rows (..., FRAME_DIM). V2 joints mirror as a pure permutation."""
    out = frames.copy()
    F = FIELD_SLICES
    out[..., F["joint_pos"]] = robot.mirror_joints(frames[..., F["joint_pos"]])
    out[..., F["joint_vel"]] = robot.mirror_joints(frames[..., F["joint_vel"]])
    out[..., F["projected_gravity"]] = frames[..., F["projected_gravity"]] * np.array([1.0, -1.0, 1.0])
    out[..., F["base_lin_vel_b"]] = frames[..., F["base_lin_vel_b"]] * np.array([1.0, -1.0, 1.0])
    out[..., F["base_ang_vel_b"]] = frames[..., F["base_ang_vel_b"]] * np.array([-1.0, 1.0, -1.0])
    fp = frames[..., F["foot_pos_b"]].reshape(frames.shape[:-1] + (2, 3))[..., ::-1, :] * np.array([1.0, -1.0, 1.0])
    out[..., F["foot_pos_b"]] = fp.reshape(frames.shape[:-1] + (6,))
    out[..., F["foot_height"]] = frames[..., F["foot_height"]][..., ::-1]
    out[..., F["contact"]] = frames[..., F["contact"]][..., ::-1]
    return out

WARMUP_CYCLES = 3
ROLLOVER_PASSES = 2              # rocker profile from the flat-foot CoM plan, then once more on its own
IK_RETRY_RESIDUAL = 5e-4         # m; a frame that ends further than this from its targets gets more iterations


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """wxyz quaternion for intrinsic z-y-x (yaw, then pitch, then roll)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def mat_from_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat_from_euler(0.0, pitch, yaw))
    return R.reshape(3, 3)


@dataclass
class GaitCycle:
    cmd: np.ndarray
    period: float
    dt: float                     # spacing of the frames: period / n_frames (phase-normalised)
    frames: np.ndarray            # (N, FRAME_DIM) table rows
    qpos: np.ndarray              # (N, nq) for display, cycle-local world frame
    sole_pos: dict                # side -> (N, 3)
    com: np.ndarray               # (N, 3) target CoM
    zmp_ref: np.ndarray           # (N, 2)
    zmp: np.ndarray               # (N, 2) ZMP implied by the final CoM motion
    support: list                 # N polygons (M,2)
    contact: np.ndarray           # (N, 2) [left, right]
    ik_residual: np.ndarray       # (N, 2) max [position m, rotation rad] task error
    footsteps: list = field(default_factory=list)   # [(side, xy, yaw)] near the cycle
    qpos_ext: np.ndarray | None = None  # (N+2, nq) one extra frame each side, for derivatives
    style: dict = field(default_factory=dict)
    path: np.ndarray | None = None      # (N, 3) commanded path pose [x, y, yaw], cycle-local
    standing: bool = False
    pitch_profiles: dict | None = None  # side -> (u, pitch) the rocker model gave the stance feet

    @property
    def n_frames(self) -> int:
        return self.frames.shape[0]

    @property
    def freq(self) -> float:
        """Gait clock frequency [cycles/s]."""
        return 1.0 / self.period

    def field(self, name: str) -> np.ndarray:
        return self.frames[:, FIELD_SLICES[name]]


class GaitGenerator:
    def __init__(self, robot: RobotModel):
        self.r = robot
        self.ik = WholeBodyIK(robot)
        self._corners = {s: robot.sole_corners_local(s) for s in SIDES}
        self._stand_cache: dict = {}
        self._rollover_cache: dict = {}
        self._warm_q: np.ndarray | None = None

    # ------------------------------------------------------------------ helpers
    def period(self, cmd, st: dict) -> float:
        """Cycle length for a command from the cadence law (cadence.py)."""
        return period_for(cmd, st, self.r.leg_length, self.r.command_ranges["vx"][1])

    def _plan(self, cmd, st: dict) -> GaitPlan:
        sl = self.r.soles["left"]
        return GaitPlan(cmd, st, sl.half_width, sl.front, sl.back, self.period(cmd, st), self.r.n_phase)

    def _twist_amplitude(self, cmd, st: dict, period: float) -> float:
        """Pelvis yaw amplitude [rad]: the knob is its value at max forward speed, scaled by the
        forward step length, since pelvic rotation follows sagittal step length (Huang et al.
        2011) and is absent stepping in place or sideways."""
        vx = float(cmd[0])
        lo, hi = self.r.command_ranges["vx"]
        v_ref = hi if vx >= 0.0 else lo
        if abs(v_ref) < 1e-9:
            return 0.0
        s_ref = abs(v_ref) * self.period((v_ref, 0.0, 0.0), st) / 2.0
        frac = float(np.clip(abs(vx) * period / 2.0 / s_ref, 0.0, 1.5))
        return math.radians(st["twist"]) * frac

    def _sole_targets(self, plan: GaitPlan, t: float):
        pos, mat, contact = {}, {}, {}
        for side in SIDES:
            p, yaw, pitch, c = plan.foot_state(side, t)
            pos[side] = p
            mat[side] = mat_from_yaw_pitch(yaw, pitch)
            contact[side] = c
        return pos, mat, contact

    def _initial_q(self, height: float) -> np.ndarray:
        if self._warm_q is not None:
            return self._warm_q.copy()
        # A bent-knee seed keeps the solver away from the straight-knee singularity.
        j = bent_knee_pose(self.r)
        return self.r.make_qpos(np.array([0, 0, height]), np.array([1.0, 0, 0, 0]), j)

    # ------------------------------------------------------------------ stand
    def stand(self, style: Style) -> tuple[np.ndarray, np.ndarray]:
        """Standing pose for the style's zero-command knobs. Returns (frame row, qpos)."""
        st = style.resolve((0.0, 0.0, 0.0), self.r.command_ranges)
        key = tuple(sorted(st.items()))
        if key in self._stand_cache:
            return self._stand_cache[key]
        w = st["stance_width"]
        toe = math.radians(st["toe_out"])
        pos = {"left": np.array([0.0, w / 2, 0.0]), "right": np.array([0.0, -w / 2, 0.0])}
        mat = {"left": mat_from_yaw_pitch(toe, 0.0), "right": mat_from_yaw_pitch(-toe, 0.0)}
        tg = IKTargets(com=np.array([st["com_forward"], 0.0, st["com_height"]]),
                       base_quat=quat_from_euler(0.0, math.radians(st["trunk_pitch"]), 0.0),
                       sole_pos=pos, sole_mat=mat)
        q, _ = self.ik.solve(self._initial_q(self.r.straight_base_height - 0.02), tg, iters=200)
        frame = self._frames_from_qpos(np.stack([q, q, q]), np.array([[1.0, 1.0]]), self.r.control_dt)[0]
        self._stand_cache[key] = (frame, q)
        return frame, q

    def stand_cycle(self, style: Style) -> GaitCycle:
        """The standing pose as a constant 'cycle', so display and analysis need no special case."""
        frame, q = self.stand(style)
        st = style.resolve((0.0, 0.0, 0.0), self.r.command_ranges)
        plan = self._plan((0.0, 0.0, 0.0), st)
        n = plan.n_frames
        r = self.r
        r.forward(q)
        soles = {s: r.d.site_xpos[r.sole_site_ids[s]].copy() for s in SIDES}
        com = r.d.subtree_com[r.base_id].copy()
        rects = []
        for s in SIDES:
            R = r.d.site_xmat[r.sole_site_ids[s]].reshape(3, 3)
            rects.append(soles[s][:2] + self._corners[s] @ R[:2, :2].T)
        support = convex_hull(np.concatenate(rects))
        return GaitCycle(
            cmd=np.zeros(3), period=plan.period, dt=plan.period / n,
            frames=np.repeat(frame[None], n, axis=0),
            qpos=np.repeat(q[None], n, axis=0), qpos_ext=np.repeat(q[None], n + 2, axis=0),
            sole_pos={s: np.repeat(soles[s][None], n, axis=0) for s in SIDES},
            com=np.repeat(com[None], n, axis=0),
            zmp_ref=np.repeat(com[None, :2], n, axis=0), zmp=np.repeat(com[None, :2], n, axis=0),
            support=[support] * n, contact=np.ones((n, 2)), ik_residual=np.zeros((n, 2)),
            footsteps=[("left", soles["left"][:2], math.radians(st["toe_out"])),
                       ("right", soles["right"][:2], -math.radians(st["toe_out"]))],
            style=st, path=np.zeros((n, 3)), standing=True,
        )

    # ------------------------------------------------------------------ cycle
    def _rollover(self, style: Style) -> Rollover:
        _, q_stand = self.stand(style)
        key = q_stand.tobytes()
        if key not in self._rollover_cache:
            self._rollover_cache[key] = Rollover(self.r, q_stand)
        return self._rollover_cache[key]

    def _body_targets(self, plan: GaitPlan, st: dict, cmd: np.ndarray):
        """CoM and base-orientation targets as a function of time, for this plan's ZMP reference."""
        T, N = plan.period, plan.n_frames
        dt = T / N
        t0 = WARMUP_CYCLES * T

        # --- CoM: preview control on the ZMP reference, from just before the path start.
        horizon = 1.2
        t_end = t0 + T + 2 * dt + horizon + 0.05
        ts = np.arange(-T, t_end, PLANNER_DT)
        zref = np.array([plan.zmp_ref(t) for t in ts])
        p0, yaw0 = plan.path(ts[0])
        v0 = rotz(yaw0) @ np.array([plan.vx, plan.vy])
        x0 = np.array([[p0[0], v0[0], 0.0], [p0[1], v0[1], 0.0]])
        com_xy = preview_com(zref, st["com_height"], PLANNER_DT, x0=x0, horizon=horizon)
        twist = self._twist_amplitude(cmd, st, T)

        def body_at(t: float):
            phase = (t / T) % 1.0
            # "single-support centred" phase angle: sin peaks mid right-single-support
            psi = 2.0 * np.pi * (phase - st["double_support"] / 4.0)
            c = sample_com(com_xy, ts[0], t)
            pth, yaw = plan.path(t)
            Rp = rotz(yaw)
            local = Rp.T @ (c - pth)
            local[1] *= st["sway_gain"]
            local[0] += st["com_forward"]
            cxy = pth + Rp @ local
            cz = st["com_height"] + st["bob_amplitude"] * math.cos(4.0 * np.pi * phase + math.radians(st["bob_phase"]))
            roll = math.radians(st["trunk_roll_sway"]) * math.sin(psi) - math.radians(st["trunk_lean"])
            pitch = math.radians(st["trunk_pitch"])
            # Pelvis yaw peaks at heel strike: + puts the hip of the foot landing at phase 0
            # (right) forward then, and the left one at phase 0.5.
            yaw_b = yaw + twist * math.cos(2.0 * np.pi * phase)
            return np.array([cxy[0], cxy[1], cz]), quat_from_euler(roll, pitch, yaw_b)

        return body_at, t0, dt

    def _solve(self, plan: GaitPlan, st: dict, style: Style, cmd: np.ndarray) -> dict:
        """CoM plan + whole-body IK over one cycle (plus one frame each side) for a plan."""
        body_at, t0, dt = self._body_targets(plan, st, cmd)
        t_out = t0 + dt * np.arange(-1, plan.n_frames + 1)   # one extra frame each side for derivatives
        targets = [body_at(t) for t in t_out]
        com_t = [c for c, _ in targets]
        base_q = [q for _, q in targets]

        # --- IK, warm-started along the timeline
        _, q_stand = self.stand(style)
        q = q_stand.copy()
        pth, yaw = plan.path(t_out[0])
        q[0:2] = pth
        q[3:7] = quat_from_euler(0.0, 0.0, yaw)
        qs, residual, contacts, soles = [], [], [], {s: [] for s in SIDES}
        for i, t in enumerate(t_out):
            pos, mat, contact = self._sole_targets(plan, t)
            tg = IKTargets(com=com_t[i], base_quat=base_q[i], sole_pos=pos, sole_mat=mat, contact=contact)
            q, e = self.ik.solve(q, tg, iters=60 if i == 0 else 25)
            res = self.ik.split_residual(e, contact)
            if res[0] > IK_RETRY_RESIDUAL:
                # Near a straight knee (long single support, big sway) the warm start can stall
                # in a local minimum short of a reachable target. Retry longer, from there and
                # from the standing legs, and keep whichever lands closer.
                seed = q.copy()
                seed[self.r.qpos_adr] = q_stand[self.r.qpos_adr]
                for q_try, e_try in (self.ik.solve(q, tg, iters=150), self.ik.solve(seed, tg, iters=150)):
                    r_try = self.ik.split_residual(e_try, contact)
                    if r_try[0] < res[0]:
                        q, e, res = q_try, e_try, r_try
            qs.append(q.copy())
            residual.append(res)
            contacts.append([float(contact["left"]), float(contact["right"])])
            for s in SIDES:
                soles[s].append(pos[s])
        return {"t_out": t_out, "t0": t0, "dt": dt, "qs": np.array(qs), "com_t": com_t,
                "residual": residual, "contacts": np.array(contacts), "soles": soles}

    def cycle(self, style: Style, cmd) -> GaitCycle:
        cmd = np.asarray(cmd, dtype=float)
        st = style.resolve(cmd, self.r.command_ranges)
        plan = self._plan(cmd, st)
        if st["rollover_gain"] > 0.0:
            # Rocker model: foot pitch from the leg geometry the plan implies (rollover.py),
            # first on the flat-foot CoM plan, then on the plan its own ZMP reference gives.
            ro = self._rollover(style)
            t_ref = WARMUP_CYCLES * plan.period
            for _ in range(ROLLOVER_PASSES):
                body_at, _, _ = self._body_targets(plan, st, cmd)
                plan.set_pitch_profiles(ro.profiles(plan, st, body_at, t_ref))
        sol = self._solve(plan, st, style, cmd)
        T, dt, t0, t_out = plan.period, sol["dt"], sol["t0"], sol["t_out"]
        qs, contacts, soles, residual = sol["qs"], sol["contacts"], sol["soles"], sol["residual"]

        frames = self._frames_from_qpos(qs, contacts[1:-1], dt)

        # --- express display data in the cycle-local frame (path pose at t0 -> origin)
        p_ref, yaw_ref = plan.path(t0)
        R0 = rotz(-yaw_ref)
        loc = lambda xy: (np.asarray(xy) - p_ref) @ R0.T
        qpos_loc = qs.copy()
        qpos_loc[:, 0:2] = loc(qs[:, 0:2])
        qz = quat_from_euler(0.0, 0.0, -yaw_ref)
        for i in range(len(qpos_loc)):
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, qz, qs[i, 3:7])
            qpos_loc[i, 3:7] = out
        com_arr = np.array(sol["com_t"])
        com_loc = np.column_stack([loc(com_arr[:, :2]), com_arr[:, 2]])

        # ZMP implied by the final (styled) CoM: p = c_xy - c_z * a_xy / (g + a_z)
        acc = (com_arr[2:] - 2 * com_arr[1:-1] + com_arr[:-2]) / (dt * dt)
        zmp = com_arr[1:-1, :2] - com_arr[1:-1, 2:3] * acc[:, :2] / (G + acc[:, 2:3])
        support = [plan.support_polygon(t, self._corners) for t in t_out[1:-1]]

        k0 = plan.step_index(t0)
        steps = [plan.footstep(k) for k in range(k0 - 2, k0 + 4)]

        return GaitCycle(
            cmd=cmd, period=T, dt=dt, frames=frames,
            qpos=qpos_loc[1:-1], qpos_ext=qpos_loc,
            sole_pos={s: np.column_stack([loc(np.array(soles[s])[1:-1, :2]), np.array(soles[s])[1:-1, 2]]) for s in SIDES},
            com=com_loc[1:-1],
            zmp_ref=loc(np.array([plan.zmp_ref(t) for t in t_out[1:-1]])),
            zmp=loc(zmp),
            support=[loc(p) for p in support],
            contact=contacts[1:-1],
            ik_residual=np.array(residual[1:-1]),
            footsteps=[(f.side, loc(f.pos), f.yaw - yaw_ref) for f in steps],
            path=np.array([np.r_[loc(plan.path(t)[0]), plan.path(t)[1] - yaw_ref] for t in t_out[1:-1]]),
            style=st,
            pitch_profiles=plan.profiles,
        )

    # ------------------------------------------------------------------ table rows
    def _frames_from_qpos(self, qs: np.ndarray, contacts: np.ndarray, dt: float) -> np.ndarray:
        """Table rows for qs[1:-1]; qs carries one extra frame on each side."""
        m, r = self.r.m, self.r
        n = qs.shape[0] - 2
        out = np.zeros((n, FRAME_DIM))
        qvel = np.zeros(m.nv)
        R = np.zeros(9)
        for i in range(n):
            q = qs[i + 1]
            mujoco.mj_differentiatePos(m, qvel, 2.0 * dt, qs[i], qs[i + 2])
            r.forward(q)
            mujoco.mju_quat2Mat(R, q[3:7])
            Rb = R.reshape(3, 3)
            base = q[0:3]
            row = out[i]
            row[FIELD_SLICES["joint_pos"]] = q[r.qpos_adr]
            row[FIELD_SLICES["joint_vel"]] = qvel[r.dof_adr]
            row[FIELD_SLICES["base_height"]] = base[2]
            row[FIELD_SLICES["projected_gravity"]] = Rb.T @ np.array([0.0, 0.0, -1.0])
            row[FIELD_SLICES["base_lin_vel_b"]] = Rb.T @ qvel[0:3]   # free-joint linvel is world frame
            row[FIELD_SLICES["base_ang_vel_b"]] = qvel[3:6]          # free-joint angvel is body frame
            feet = [Rb.T @ (r.d.xpos[r.foot_body_ids[s]] - base) for s in SIDES]
            row[FIELD_SLICES["foot_pos_b"]] = np.concatenate(feet)
            row[FIELD_SLICES["foot_height"]] = [r.d.site_xpos[r.sole_site_ids[s]][2] for s in SIDES]
            row[FIELD_SLICES["contact"]] = contacts[min(i, len(contacts) - 1)]
        return out
