"""Whole-body damped least-squares IK on the floating-base model.

Tasks: both sole poses (6 each), whole-body CoM (3), base orientation (3) = 18 rows for the
18 DoF (6 floating + 12 joints), plus a weak posture prior.

When the targets are unreachable (leg fully stretched, ankle on its limit) the weights decide
what gives first. Stance feet and CoM xy are kept hard because they define balance and
contact; CoM height, swing-foot tilt and torso orientation are soft. Joint limits are
handled with a small active set, so a joint pinned at a limit drops out and the rest keep
solving instead of fighting a clamp.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .robot import SIDES, RobotModel

DEFAULT_WEIGHTS = {
    "stance_pos": 1.0,
    "stance_rot": 0.2,
    "swing_pos": 1.0,
    "swing_rot": 0.002,   # ~1 cm of foot error trades for ~0.2 rad of tilt
    "com_xy": 1.0,
    "com_z": 0.15,
    "base_rot": 0.05,
}


@dataclass
class IKTargets:
    com: np.ndarray                 # (3,)
    base_quat: np.ndarray           # (4,) wxyz
    sole_pos: dict                  # side -> (3,)
    sole_mat: dict                  # side -> (3,3)
    contact: dict = field(default_factory=lambda: {s: True for s in SIDES})


def _rot_error(R_target: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking R_cur to R_target."""
    R_err = R_target @ R_cur.T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R_err.ravel())
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def bent_knee_pose(robot: RobotModel, knee: float = -0.8) -> np.ndarray:
    j = np.zeros(robot.num_joints)
    for i, n in enumerate(robot.joint_names):
        if "knee" in n:
            j[i] = knee
        elif "hip_pitch" in n or "ankle_pitch" in n:
            j[i] = knee / 2.0
    return j


class WholeBodyIK:
    def __init__(self, robot: RobotModel, weights: dict | None = None, w_posture=1e-6, damping=1e-6):
        self.r = robot
        self.m = robot.m
        self.d = mujoco.MjData(robot.m)
        nv = self.m.nv
        self._jacp = np.zeros((3, nv))
        self._jacr = np.zeros((3, nv))
        self.weights = dict(DEFAULT_WEIGHTS, **(weights or {}))
        self.w_posture = w_posture
        self.damping = damping
        # Weak prior toward a bent-knee stance: only decides directions the tasks leave free
        # (and keeps the knee off the straight-leg singularity).
        self.nominal = bent_knee_pose(robot)

    def _forward(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def _row_weights(self, tg: IKTargets) -> np.ndarray:
        w = self.weights
        rows = []
        for side in SIDES:
            if tg.contact[side]:
                rows += [w["stance_pos"]] * 3 + [w["stance_rot"]] * 3
            else:
                rows += [w["swing_pos"]] * 3 + [w["swing_rot"]] * 3
        rows += [w["com_xy"]] * 2 + [w["com_z"]] + [w["base_rot"]] * 3
        return np.array(rows)

    def _residual_and_jacobian(self, tg: IKTargets):
        m, d = self.m, self.d
        rows_e, rows_J = [], []
        for side in SIDES:
            sid = self.r.sole_site_ids[side]
            mujoco.mj_jacSite(m, d, self._jacp, self._jacr, sid)
            rows_e.append(tg.sole_pos[side] - d.site_xpos[sid])
            rows_e.append(_rot_error(tg.sole_mat[side], d.site_xmat[sid].reshape(3, 3)))
            rows_J += [self._jacp.copy(), self._jacr.copy()]
        mujoco.mj_jacSubtreeCom(m, d, self._jacp, self.r.base_id)
        rows_e.append(tg.com - d.subtree_com[self.r.base_id])
        rows_J.append(self._jacp.copy())
        mujoco.mj_jacBody(m, d, None, self._jacr, self.r.base_id)
        R_base = d.xmat[self.r.base_id].reshape(3, 3)
        R_t = np.zeros(9)
        mujoco.mju_quat2Mat(R_t, tg.base_quat)
        rows_e.append(_rot_error(R_t.reshape(3, 3), R_base))
        rows_J.append(self._jacr.copy())
        return np.concatenate(rows_e), np.vstack(rows_J)

    def solve(self, q0: np.ndarray, tg: IKTargets, iters: int = 30, tol: float = 1e-7):
        """Return (qpos, residual vector)."""
        q = q0.copy()
        nv = self.m.nv
        dof = self.r.dof_adr
        lo, hi = self.r.lower, self.r.upper
        W = self._row_weights(tg)
        prev_cost = np.inf
        for _ in range(iters):
            self._forward(q)
            e, J = self._residual_and_jacobian(tg)
            cost = float(np.dot(W * e, e))
            if float(np.dot(e, e)) < tol * tol or abs(prev_cost - cost) < 1e-14:
                break
            prev_cost = cost
            H = J.T @ (W[:, None] * J)
            g = J.T @ (W * e)
            H[dof, dof] += self.w_posture
            g[dof] += self.w_posture * (self.nominal - q[self.r.qpos_adr])
            # Levenberg-Marquardt style damping grows with the error, which tames the steps
            # near the straight-knee singularity without slowing convergence when close.
            H[np.diag_indices(nv)] += self.damping + 1e-3 * cost

            # Active set on the joint limits: a joint sitting on a limit and being pushed
            # outward is frozen and the step is re-solved without it.
            qj = q[self.r.qpos_adr]
            free = np.ones(nv, dtype=bool)
            for _ in range(4):
                idx = np.flatnonzero(free)
                dq = np.zeros(nv)
                dq[idx] = np.linalg.solve(H[np.ix_(idx, idx)], g[idx])
                d_j = dq[dof]
                push = ((qj <= lo + 1e-9) & (d_j < 0)) | ((qj >= hi - 1e-9) & (d_j > 0))
                push &= free[dof]
                if not push.any():
                    break
                free[dof[push]] = False
            # Never step past a limit: shorten each joint step to stay inside the box.
            d_j = dq[dof]
            room = np.where(d_j > 0, hi - qj, lo - qj)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(np.abs(d_j) > 1e-12, room / d_j, np.inf)
            dq[dof] = np.where(ratio < 1.0, d_j * np.clip(ratio, 0.0, 1.0), d_j)
            mujoco.mj_integratePos(self.m, q, dq, 1.0)
            q[self.r.qpos_adr] = np.clip(q[self.r.qpos_adr], lo, hi)
        self._forward(q)
        e, _ = self._residual_and_jacobian(tg)
        return q, e

    @staticmethod
    def split_residual(e: np.ndarray, contact: dict | None = None) -> tuple[float, float]:
        """(max foot/CoM-xy position error [m], max stance-foot rotation error [rad]).

        Soft tasks (CoM height, swing tilt, torso orientation) are excluded: missing them
        changes the style, not whether the motion is valid.
        """
        contact = contact or {s: True for s in SIDES}
        pos = [e[0:3], e[6:9], e[12:14]]
        rot = [e[3:6] if contact["left"] else np.zeros(3), e[9:12] if contact["right"] else np.zeros(3)]
        return float(np.max(np.abs(np.concatenate(pos)))), float(np.max(np.abs(np.concatenate(rot))))
