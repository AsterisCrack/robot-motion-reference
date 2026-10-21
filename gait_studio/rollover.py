"""Heel-to-toe from the rocker model: the stance foot's pitch follows the shank.

Human stance has three rockers (Perry):
  * heel rocker: the leg arrives inclined while the ankle is near neutral, so the foot lands
    toes-up by the shank's lean and is lowered to flat about the heel (loading response);
  * ankle rocker: the foot is flat and the shank rolls over it, the ankle dorsiflexing;
  * forefoot rocker: once the ankle has used its range (~10 deg past neutral) the heel lifts
    and the foot pivots on the toe; push-off plantarflexes it further before toe off.
Kinematically the ankle angle is foot pitch minus shank lean, so none of this needs a hand-set
angle: the pitch emerges from where the leg is. It grows with step length, vanishes when
stepping in place or sideways (no sagittal shank lean in the foot's frame) and reverses walking
backward (toe strike, heel last). Robots add a second trigger (Kajita; Miura et al. 2011,
HRP-4C): when the hip is further from the ankle than the leg reaches, the foot must pivot on its
edge so the ankle rises toward the hip. Why bother: a rolling foot smooths the step-to-step
transition, cutting heel-strike collision losses (Adamczyk, Collins & Kuo 2006), and lengthens
the effective leg.

The leg is evaluated from the plan, not from an IK solve: the hip-pitch axis from the CoM target
and base orientation (its offset from the CoM measured standing), the ankle-pitch axis from the
planned flat footstep (rigid in the foot), and the knee from the thigh/shank lengths. An IK
solve would sag wherever a near-straight leg can't reach a step flat-footed, and feeding that
back makes the pitch chase its own errors. gait.py builds the profiles on the flat-foot CoM
plan, refines them once on the plan their own ZMP reference gives, then solves the IK.
"""
from __future__ import annotations

import math
from typing import Callable

import mujoco
import numpy as np

from .footsteps import GaitPlan, rotz, smoothstep
from .robot import SIDES, RobotModel

N_PROFILE = 41              # samples over one stance
MAX_PITCH = math.radians(45.0)
SOFT_WIDTH = math.radians(1.0)   # softplus width at the ankle-range limit: no velocity kink at heel off

BodyAt = Callable[[float], tuple]   # t -> (CoM target (3,), base quaternion wxyz (4,))


def _softplus(x: np.ndarray, w: float = SOFT_WIDTH) -> np.ndarray:
    return w * np.logaddexp(0.0, x / w)


def _flat_axes(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(forward, lateral, up) of a body's flat frame: its heading, world up."""
    f = np.array([R[0, 0], R[1, 0], 0.0])
    f /= max(float(np.linalg.norm(f)), 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    return f, np.cross(up, f), up


class Rollover:
    """Per-robot constants of the rocker model, measured once from the standing pose."""

    def __init__(self, robot: RobotModel, q_stand: np.ndarray):
        self.r = robot
        d = robot.forward(q_stand)
        com = d.subtree_com[robot.base_id].copy()
        Rb = d.xmat[robot.base_id].reshape(3, 3)
        self.hip_c, self.ankle_f, self.l_thigh, self.l_shank = {}, {}, {}, {}
        self.edges, self.reach_stand = {}, {}
        for s in SIDES:
            ids = robot.leg_joint_ids[s]
            hip, knee, ankle = (d.xanchor[ids[k]].copy() for k in ("hip_pitch", "knee", "ankle_pitch"))
            sole = d.site_xpos[robot.sole_site_ids[s]].copy()
            f, lat, up = _flat_axes(d.site_xmat[robot.sole_site_ids[s]].reshape(3, 3))
            # Hip-pitch axis relative to the CoM in the base frame (the legs move it a few mm).
            self.hip_c[s] = Rb.T @ (hip - com)
            # Ankle-pitch axis relative to the sole centre in the foot frame: rigid, exact.
            a = ankle - sole
            self.ankle_f[s] = np.array([a @ f, a @ lat, a @ up])
            self.l_thigh[s] = float(np.linalg.norm(hip - knee))
            self.l_shank[s] = float(np.linalg.norm(knee - ankle))
            # Toe and heel edges relative to the ankle-pitch axis, foot frame (x fwd, z up).
            sl = robot.soles[s]
            cx, cz = -self.ankle_f[s][0], -self.ankle_f[s][2]
            self.edges[s] = {"toe": np.array([cx + sl.front, cz]), "heel": np.array([cx - sl.back, cz])}
            # Hip-ankle distance when standing: the foot is flat there by definition, so the
            # reach trigger never asks for less (a near-straight style stands close to full reach).
            self.reach_stand[s] = float(np.linalg.norm(hip - ankle))

    # ------------------------------------------------------------------ leg from the plan
    def leg(self, side: str, plan: GaitPlan, k: int, t: float, body_at: BodyAt) -> tuple[float, np.ndarray]:
        """(shank lean, hip relative to the ankle (x fwd, z up)) for flat footstep k at time t.

        Shank lean: toward the toes, in the foot's sagittal plane, + = knee ahead of the ankle.
        It is the ankle angle the leg asks of a flat foot.
        """
        f_step = plan.footstep(k)
        R2 = rotz(f_step.yaw)
        fwd = np.array([R2[0, 0], R2[1, 0], 0.0])
        lat = np.array([R2[0, 1], R2[1, 1], 0.0])
        up = np.array([0.0, 0.0, 1.0])
        a = self.ankle_f[side]
        ankle = np.array([f_step.pos[0], f_step.pos[1], 0.0]) + a[0] * fwd + a[1] * lat + a[2] * up
        com, quat = body_at(t)
        Rb = np.zeros(9)
        mujoco.mju_quat2Mat(Rb, quat)
        hip = com + Rb.reshape(3, 3) @ self.hip_c[side]
        h = hip - ankle
        hx, hz = float(h @ fwd), float(h @ up)
        # Knee: two links from the ankle to the hip, bending forward in the foot's sagittal plane.
        dist = math.hypot(hx, hz)
        l1, l2 = self.l_thigh[side], self.l_shank[side]
        c = (l2 * l2 + dist * dist - l1 * l1) / (2.0 * l2 * max(dist, 1e-9))
        beta = math.acos(float(np.clip(c, -1.0, 1.0)))
        leg_dir = math.atan2(hx, hz)                    # leg line from vertical, + = hip ahead
        return leg_dir + beta, np.array([hx, hz])       # knee forward of the leg line

    # ------------------------------------------------------------------ reach trigger
    def reach_pitch(self, side: str, hip: np.ndarray, reach: float) -> float:
        """Pitch (+ toes down) the flat stance foot needs so the hip-ankle distance <= reach.

        The foot pivots on the edge on the hip's side (toe if the hip is ahead), which swings
        the ankle up toward the hip: ankle(p) = edge + R_y(p) (ankle - edge), ankle at the origin.
        Only the hip's fore-aft offset can call for it: with the hip straight above the ankle,
        pivoting on either edge would not help, so the limit is at least the hip's height. That
        also makes the demand fade to zero (no heel -> toe jump) as the hip passes over the ankle.
        """
        reach = max(reach, abs(float(hip[1])) + 1e-4)
        if float(np.hypot(*hip)) <= reach:
            return 0.0
        sign = 1.0 if hip[0] >= 0.0 else -1.0
        edge = self.edges[side]["toe" if sign > 0 else "heel"]

        def dist(mag: float) -> float:
            p = sign * mag
            c, s = math.cos(p), math.sin(p)
            v = -edge
            ankle = edge + np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])
            return float(np.hypot(*(hip - ankle)))

        lo, hi = 0.0, MAX_PITCH
        if dist(hi) > reach:            # can't get there: take the closest the pivot allows
            mags = np.linspace(0.0, MAX_PITCH, 46)
            return sign * float(mags[int(np.argmin([dist(m) for m in mags]))])
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if dist(mid) > reach else (lo, mid)
        return sign * hi

    # ------------------------------------------------------------------ profiles
    @staticmethod
    def _neutral(lean: np.ndarray, hip_x: np.ndarray) -> float:
        """Shank lean at mid-stance, when the hip passes over the ankle.

        This is the ankle's neutral for the rocker rules. A human's neutral (foot square to the
        shank) is where the straight leg stands over the foot. This robot stands crouched, and
        its knee straightens toward both ends of the stance; measured from the crouched standing
        pose, that would bias every step toward a bigger heel strike and a smaller toe off.

        Taken as the mean lean over the stance: walking, the hip spans the ankle about evenly,
        so the mean is the mid-stance value. When stepping in place or turning the hip may never
        pass over the ankle (each foot sees it travel a few cm one way); picking a crossing or the
        end of that range would give one foot of a turn all the pitch and the other none. The
        mean splits it evenly and stays continuous across commands.
        """
        return float(lean.mean())

    def profiles(self, plan: GaitPlan, st: dict, body_at: BodyAt, t_ref: float) -> dict:
        """side -> (u grid, pitch over the stance), + = toes down. t_ref: a right touchdown."""
        gain = float(st["rollover_gain"])
        u = np.linspace(0.0, 1.0, N_PROFILE)
        h = math.radians(st["heel_off_angle"])
        contact = math.radians(st["contact_ankle"])
        push = math.radians(st["push_off"])
        reach_knob = float(st["max_leg_reach"]) * self.r.hip_ankle_length
        k_right = plan.step_index(t_ref)
        if k_right % 2:
            k_right += 1
        out = {}
        for side in SIDES:
            k = k_right if side == "right" else k_right + 1
            legs = [self.leg(side, plan, k, k * plan.Ts + x * plan.Tst, body_at) for x in u]
            lean = np.array([g[0] for g in legs])
            hip = np.array([g[1] for g in legs])
            lean = lean - self._neutral(lean, hip[:, 0])

            # Heel rocker: land with the ankle at `contact_ankle`, i.e. pitched by the shank's lean,
            # then lower the foot over the loading response. (+ lean at touchdown: toe strike.)
            p_td = lean[0] - contact
            p_heel = p_td * (1.0 - np.array([smoothstep(x) for x in u / max(st["heel_roll_time"], 1e-3)]))
            # Ankle range: the foot stays flat only while the shank leans less than h either way.
            down = _softplus(lean - h)             # shank far over the toes -> heel lifts (toe pivot)
            up = _softplus(-lean - h)              # shank far back -> toes lift (heel pivot)
            # Reach: the hip further than the leg reaches -> pivot toward the hip.
            reach = max(reach_knob, self.reach_stand[side])
            r = np.array([self.reach_pitch(side, hip[i], reach) for i in range(len(u))])
            down = np.maximum.reduce([down, np.maximum(p_heel, 0.0), np.maximum(r, 0.0)])
            up = np.maximum.reduce([up, np.maximum(-p_heel, 0.0), np.maximum(-r, 0.0)])
            pitch = down - up
            # Push-off through the final double support, toward the side the foot leaves from.
            u_ps = 1.0 - plan.Tds / plan.Tst
            if push > 0.0 and u_ps < 1.0:
                ramp = np.array([smoothstep(x) for x in (u - u_ps) / (1.0 - u_ps)])
                pitch = pitch + push * ramp * float(np.clip(lean[-1] / max(h, 1e-3), -1.0, 1.0))
            out[side] = (u, np.clip(gain * pitch, -MAX_PITCH, MAX_PITCH))
        return out
