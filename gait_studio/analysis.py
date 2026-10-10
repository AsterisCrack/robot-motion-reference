"""Feasibility checks for a generated cycle: limits, servo speed/torque, balance, foot clearance."""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .footsteps import signed_distance_to_polygon
from .gait import GaitCycle
from .robot import SIDES, RobotModel

OK, WARN, FAIL = "ok", "warn", "fail"
_RANK = {OK: 0, WARN: 1, FAIL: 2}


@dataclass
class Check:
    name: str
    value: float
    unit: str
    status: str
    detail: str = ""


@dataclass
class Report:
    status: str
    checks: list[Check]
    torque: np.ndarray            # (N, 12) estimated joint torques
    joint_vel: np.ndarray         # (N, 12)
    zmp_margin: np.ndarray        # (N,) metres, positive inside the support polygon
    foot_gap: np.ndarray          # (N,) metres, negative = soles overlap in xy
    per_joint: dict = field(default_factory=dict)
    envelope_usage: np.ndarray | None = None   # (N, 12) |tau| / available torque at that speed

    def summary(self) -> str:
        bad = [c for c in self.checks if c.status != OK]
        if not bad:
            return "OK"
        return "; ".join(f"{c.name} {c.value:.3g}{c.unit} ({c.status})" for c in bad)


def _grade(value: float, warn: float, fail: float, higher_is_worse: bool = True) -> str:
    if higher_is_worse:
        return FAIL if value > fail else WARN if value > warn else OK
    return FAIL if value < fail else WARN if value < warn else OK


def _rect(center_xy, R, corners_local):
    return center_xy + corners_local @ R[:2, :2].T


def separation(P: np.ndarray, Q: np.ndarray) -> float:
    """Separating-axis signed separation of two convex polygons: >0 gap, <0 penetration depth."""
    best = -np.inf
    for poly in (P, Q):
        n = len(poly)
        for i in range(n):
            e = poly[(i + 1) % n] - poly[i]
            ax = np.array([-e[1], e[0]])
            L = np.linalg.norm(ax)
            if L < 1e-12:
                continue
            ax /= L
            p, q = P @ ax, Q @ ax
            gap = max(q.min() - p.max(), p.min() - q.max())
            best = max(best, gap)
    return float(best)


class Analyzer:
    def __init__(self, robot: RobotModel):
        self.r = robot
        # Separate model for inverse dynamics with every constraint off: contact forces are
        # solved for explicitly below, joint friction is not a load the motor has to carry
        # in this estimate.
        self.m = mujoco.MjModel.from_xml_string(robot.xml)
        self.m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        self.m.dof_armature[robot.dof_adr] = float(robot.armature)
        self.d = mujoco.MjData(self.m)
        self._jacp = np.zeros((3, self.m.nv))
        self._jacr = np.zeros((3, self.m.nv))
        self.corners = {s: robot.sole_corners_local(s) for s in SIDES}

    # ------------------------------------------------------------------ dynamics
    def torques(self, cyc: GaitCycle) -> np.ndarray:
        m, d, r = self.m, self.d, self.r
        qs = cyc.qpos_ext
        dt = cyc.dt
        n = cyc.n_frames
        tau = np.zeros((n, r.num_joints))
        v_half = np.zeros((len(qs) - 1, m.nv))
        for i in range(len(qs) - 1):
            mujoco.mj_differentiatePos(m, v_half[i], dt, qs[i], qs[i + 1])
        sole_xy = {s: cyc.sole_pos[s][:, :2] for s in SIDES}
        for i in range(n):
            d.qpos[:] = qs[i + 1]
            d.qvel[:] = 0.5 * (v_half[i] + v_half[i + 1])
            d.qacc[:] = (v_half[i + 1] - v_half[i]) / dt
            mujoco.mj_inverse(m, d)
            f_gen = d.qfrc_inverse.copy()

            stance = [s for s in SIDES if cyc.contact[i, SIDES.index(s)] > 0.5] or list(SIDES)
            Js = []
            for s in stance:
                mujoco.mj_jacSite(m, d, self._jacp, self._jacr, r.sole_site_ids[s])
                Js.append(np.vstack([self._jacp, self._jacr]))
            # Split the floating-base wrench over the stance feet: share by ZMP proximity.
            if len(stance) == 2:
                a, b = sole_xy["left"][i], sole_xy["right"][i]
                ab = b - a
                t = float(np.clip(np.dot(cyc.zmp[i] - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0))
                share = {"left": max(1.0 - t, 0.05), "right": max(t, 0.05)}
            else:
                share = {stance[0]: 1.0}
            A = np.hstack([J[:, :6].T for J in Js])                    # (6, 6k)
            Wd = np.concatenate([[share[s]] * 6 for s in stance])
            F = Wd * (A.T @ np.linalg.lstsq(A @ (Wd[:, None] * A.T), f_gen[:6], rcond=None)[0])
            f_contact = sum(J.T @ F[6 * k: 6 * k + 6] for k, J in enumerate(Js))
            tau[i] = (f_gen - f_contact)[r.dof_adr]
        return tau

    # ------------------------------------------------------------------ report
    def analyze(self, cyc: GaitCycle) -> Report:
        r = self.r
        jp, jv = cyc.field("joint_pos"), cyc.field("joint_vel")
        checks: list[Check] = []

        margin = np.minimum(jp - r.lower, r.upper - jp).min(axis=0)
        worst = int(np.argmin(margin))
        checks.append(Check("joint limit margin", float(np.degrees(margin[worst])), "deg",
                            _grade(float(np.degrees(margin[worst])), 2.0, -0.1, higher_is_worse=False),
                            r.joint_names[worst]))

        vmax = np.abs(jv).max(axis=0)
        worst = int(np.argmax(vmax))
        # The servo cannot outrun its back-EMF (no-load speed) nor its firmware slew limit on the target.
        act = r.actuator
        v_fail = min(act.no_load_speed, act.slew_limit)
        checks.append(Check("peak joint speed", float(vmax[worst]), "rad/s",
                            _grade(float(vmax[worst]), v_fail * 2 / 3, v_fail),
                            r.joint_names[worst]))

        # Motor torque = joint torque + friction the servo must overcome (BAM friction budget).
        tau = act.required_motor_torque(self.torques(cyc), jv)
        tmax = np.abs(tau).max(axis=0)
        worst = int(np.argmax(tmax))
        checks.append(Check("peak joint torque", float(tmax[worst]), "N-m",
                            _grade(float(tmax[worst]), 0.7 * act.stall_torque, act.stall_torque),
                            r.joint_names[worst]))
        # Torque-speed envelope: back-EMF takes kt^2/R per rad/s when driving and gives it back when braking.
        avail = act.torque_limit(tau, jv)
        usage = np.minimum(np.abs(tau) / np.maximum(avail, 1e-6), 9.99)  # capped: past the envelope
        u = float(usage.max())
        worst = int(np.unravel_index(np.argmax(usage), usage.shape)[1])
        checks.append(Check("torque-speed usage", 100.0 * u, "%", _grade(100.0 * u, 70.0, 100.0),
                            r.joint_names[worst]))

        zm = np.array([signed_distance_to_polygon(cyc.zmp[i], cyc.support[i]) for i in range(cyc.n_frames)])
        checks.append(Check("ZMP margin", float(zm.min() * 1000), "mm",
                            _grade(float(zm.min() * 1000), 5.0, -5.0, higher_is_worse=False)))

        gaps = np.zeros(cyc.n_frames)
        for i in range(cyc.n_frames):
            self.r.forward(cyc.qpos[i])
            rects = []
            for s in SIDES:
                p, R = self.r.sole_pose(s)
                rects.append(_rect(p[:2], R, self.corners[s]))
            gaps[i] = separation(rects[0], rects[1])
        checks.append(Check("min foot gap", float(gaps.min() * 1000), "mm",
                            _grade(float(gaps.min() * 1000), 5.0, 0.0, higher_is_worse=False)))

        pos_err = float(cyc.ik_residual[:, 0].max() * 1000)
        rot_err = float(np.degrees(cyc.ik_residual[:, 1].max()))
        checks.append(Check("IK foot/CoM error", pos_err, "mm", _grade(pos_err, 2.0, 5.0)))
        checks.append(Check("IK stance tilt error", rot_err, "deg", _grade(rot_err, 1.0, 3.0)))

        status = max((c.status for c in checks), key=lambda s: _RANK[s])
        return Report(status=status, checks=checks, torque=tau, joint_vel=jv, zmp_margin=zm, foot_gap=gaps,
                      per_joint={"vel_max": vmax, "torque_max": tmax, "margin": margin}, envelope_usage=usage)
