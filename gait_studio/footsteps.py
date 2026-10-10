"""Path, footstep schedule, swing-foot curves and ZMP reference for one command.

Timeline convention (matches the env's gait clock, where sin(phi) > 0 means right stance):
  * normalised phase 0 is a RIGHT foot touchdown;
  * step k spans [k*Ts, (k+1)*Ts]; its stance foot is right for even k, left for odd k;
  * each step opens with a double-support window of length Tds, then single support on the
    new stance foot while the other foot swings from its previous footstep to the next one.
So [0, 0.5) of the cycle is right stance and [0.5, 1) is left stance; a mirror is a 0.5 shift.

Heel-to-toe: a stance foot does not have to stay flat. It can land pitched toes-up and roll
down onto the sole about its heel edge, then peel off about its toe edge (reversed when
walking backward). The pivot edge stays planted, the swing starts and ends exactly at those
rolled poses, and the ZMP reference follows the contact region (heel edge -> sweep -> toe
edge), so the LIPM CoM plan stays consistent with what the foot can actually push on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def rotz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def min_jerk(s):
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def eased_progress(s: float, ramp: float) -> float:
    """Swing progress with a smoothed-trapezoid velocity profile.

    Velocity ramps up over the first ``ramp`` of the swing (smoothstep), cruises, and ramps
    down over the last ``ramp``. Peak velocity is 1/(1-ramp) of the average, against 1.875
    for min-jerk, which is what lets a speed-limited servo carry a longer step.
    """
    r = float(np.clip(ramp, 0.05, 0.5))
    area = 1.0 - r
    if s <= r:
        u = s / r
        return r * (u ** 3 - 0.5 * u ** 4) / area
    if s >= 1.0 - r:
        w = (1.0 - s) / r
        return 1.0 - r * (w ** 3 - 0.5 * w ** 4) / area
    return (0.5 * r + (s - r)) / area


def _overshoot_shape(s):
    # zero at both ends, peak 1 at s = 2/3
    return 6.75 * s * s * (1.0 - s)


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


ZMP_EDGE_INSET = 0.005      # ZMP target this far inside a pivot edge
PIVOT_STRIP = 0.010         # support strip width along a pivot edge while the foot is pitched
ROLL_EPS = 1e-3             # rad; below this a foot counts as flat


@dataclass
class Footstep:
    side: str          # "left" / "right"
    pos: np.ndarray    # (2,) sole centre, world xy
    yaw: float


class GaitPlan:
    """Everything kinematic about one command except the CoM."""

    def __init__(self, cmd, style: dict, sole_half_width: float, sole_front: float, sole_back: float,
                 control_dt: float = 0.02):
        self.cmd = np.asarray(cmd, dtype=float)
        self.vx, self.vy, self.wz = self.cmd
        self.style = style

        n_frames = max(8, int(round(style["period"] / control_dt / 2.0)) * 2)  # even -> mirror is exact
        self.n_frames = n_frames
        self.period = n_frames * control_dt
        self.Ts = self.period / 2.0
        self.Tds = float(np.clip(style["double_support"], 0.0, 0.9)) * self.Ts
        self.Tsw = self.Ts - self.Tds

        self.sole_front, self.sole_back = sole_front, sole_back
        # Sidestepping makes every other step narrower by |vy|*Ts; widen the nominal stance so
        # the narrow one still keeps min_foot_gap between the inner sole edges.
        need = 2.0 * sole_half_width + style["min_foot_gap"] + abs(self.vy) * self.Ts
        self.width = max(style["stance_width"], need)
        self.toe_out = math.radians(style["toe_out"])
        self.steps: dict[int, Footstep] = {}

        # Rolling stance. Forward: land toes-up on the heel, leave heel-up off the toe.
        # Backward the foot leads with the toe, so both rolls flip.
        direction = 1.0 if self.vx >= 0.0 else -1.0
        self.Tst = self.Ts + self.Tds                       # stance duration of one foot
        self.pitch_strike = -direction * math.radians(style["heel_strike"])   # + = toes down
        self.pitch_off = direction * math.radians(style["toe_off"])
        r_in, r_out = float(style["heel_roll_time"]), float(style["toe_roll_time"])
        if r_in + r_out > 0.95:
            scale = 0.95 / (r_in + r_out)
            r_in, r_out = r_in * scale, r_out * scale
        self.roll_in, self.roll_out = r_in, r_out

    # ------------------------------------------------------------------ path
    def path(self, t: float) -> tuple[np.ndarray, float]:
        """Pose (xy, yaw) of the commanded path frame at time t (path starts at origin, yaw 0)."""
        vx, vy, wz = self.vx, self.vy, self.wz
        yaw = wz * t
        if abs(wz) < 1e-9:
            return np.array([vx * t, vy * t]), yaw
        s, c = math.sin(yaw), math.cos(yaw)
        x = (vx * s + vy * (c - 1.0)) / wz
        y = (vx * (1.0 - c) + vy * s) / wz
        return np.array([x, y]), yaw

    # ------------------------------------------------------------------ footsteps
    def side_of_step(self, k: int) -> str:
        return "right" if k % 2 == 0 else "left"

    def footstep(self, k: int) -> Footstep:
        if k not in self.steps:
            side = self.side_of_step(k)
            # Centre the footstep on the path at the middle of its stance interval
            # [k*Ts, (k+1)*Ts + Tds], so the average ZMP rides the path.
            t_mid = k * self.Ts + 0.5 * (self.Ts + self.Tds)
            p, yaw = self.path(t_mid)
            lat = 0.5 * self.width if side == "left" else -0.5 * self.width
            pos = p + rotz(yaw) @ np.array([0.0, lat])
            foot_yaw = yaw + (self.toe_out if side == "left" else -self.toe_out)
            self.steps[k] = Footstep(side, pos, foot_yaw)
        return self.steps[k]

    def phase(self, t: float) -> float:
        return (t / self.period) % 1.0

    # ------------------------------------------------------------------ feet
    def _where(self, side: str, t: float):
        """("stance", k, u) with u in [0, 1] through footstep k's stance, or ("swing", k0, k1, s)."""
        k = int(math.floor(t / self.Ts))
        tau = t - k * self.Ts
        if self.side_of_step(k) == side:
            return "stance", k, tau / self.Tst
        if tau < self.Tds or self.Tsw <= 1e-6:
            return "stance", k - 1, min((self.Ts + tau) / self.Tst, 1.0)
        return "swing", k - 1, k + 1, float(np.clip((tau - self.Tds) / self.Tsw, 0.0, 1.0))

    def stance_pitch(self, u: float) -> float:
        """Foot pitch (rad, + = toes down) at fraction u of the stance."""
        if self.roll_in > 0.0 and u < self.roll_in:
            return self.pitch_strike * (1.0 - smoothstep(u / self.roll_in))
        if self.roll_out > 0.0 and u > 1.0 - self.roll_out:
            return self.pitch_off * smoothstep((u - (1.0 - self.roll_out)) / self.roll_out)
        return 0.0

    def _rolled_pose(self, f: Footstep, pitch: float):
        """Sole-centre (pos3, yaw, pitch) of footstep f rolled by pitch about its planted edge."""
        fwd = rotz(f.yaw) @ np.array([1.0, 0.0])
        if pitch > 0.0:     # toes down: pivot on the toe edge
            edge = f.pos + self.sole_front * fwd
            xy = edge - self.sole_front * math.cos(pitch) * fwd
            z = self.sole_front * math.sin(pitch)
        elif pitch < 0.0:   # toes up: pivot on the heel edge
            edge = f.pos - self.sole_back * fwd
            xy = edge + self.sole_back * math.cos(pitch) * fwd
            z = -self.sole_back * math.sin(pitch)
        else:
            xy, z = f.pos, 0.0
        return np.array([xy[0], xy[1], z]), f.yaw, pitch

    def foot_state(self, side: str, t: float):
        """Return (pos3, yaw, pitch, in_contact) of a sole centre at time t."""
        where = self._where(side, t)
        if where[0] == "stance":
            pos, yaw, pitch = self._rolled_pose(self.footstep(where[1]), self.stance_pitch(where[2]))
            return pos, yaw, pitch, True

        _, k0, k1, s = where
        st = self.style
        p0, yaw0, pitch0 = self._rolled_pose(self.footstep(k0), self.stance_pitch(1.0))
        p1, yaw1, pitch1 = self._rolled_pose(self.footstep(k1), self.stance_pitch(0.0))
        prog = eased_progress(s, st["swing_ease"]) + st["swing_overshoot"] * _overshoot_shape(s)
        xy = p0[:2] + (p1[:2] - p0[:2]) * prog
        dyaw = (yaw1 - yaw0 + math.pi) % (2.0 * math.pi) - math.pi
        yaw = yaw0 + dyaw * min_jerk(s)
        outward = rotz(yaw) @ np.array([0.0, 1.0 if side == "left" else -1.0])
        xy = xy + st["swing_bulge"] * math.sin(math.pi * s) * outward

        apex = float(np.clip(st["apex_timing"], 0.05, 0.95))
        warp = s ** (math.log(0.5) / math.log(apex))
        z = st["step_height"] * math.sin(math.pi * warp) ** 2  # zero velocity at both ends
        pitch = pitch0 + (pitch1 - pitch0) * min_jerk(s)
        # Lift by the lowest sole edge, so a pitched foot never dips below the swing curve. At
        # s = 0 and 1 this equals the rolled stance height, so the hand-over is continuous.
        z += self.sole_front * max(0.0, math.sin(pitch)) + self.sole_back * max(0.0, -math.sin(pitch))
        return np.array([xy[0], xy[1], z]), yaw, pitch, False

    # ------------------------------------------------------------------ ZMP
    def _edge_offset(self, pitch: float) -> float:
        """Forward offset (from the sole centre) of the contact region for a roll angle.

        Any roll at all puts the foot on its edge, so the ZMP goes all the way there.
        """
        if abs(pitch) <= ROLL_EPS:
            return 0.0
        return (self.sole_front - ZMP_EDGE_INSET) if pitch > 0.0 else -(self.sole_back - ZMP_EDGE_INSET)

    def _contact_center(self, k: int, u: float) -> np.ndarray:
        """Centre of footstep k's contact region at stance fraction u (world xy)."""
        f = self.footstep(k)
        x_in, x_out = self._edge_offset(self.pitch_strike), self._edge_offset(self.pitch_off)
        if u <= self.roll_in:
            x = x_in
        elif u >= 1.0 - self.roll_out:
            x = x_out
        else:
            span = max(1.0 - self.roll_out - self.roll_in, 1e-6)
            x = x_in + (x_out - x_in) * (u - self.roll_in) / span
        return f.pos + x * (rotz(f.yaw) @ np.array([1.0, 0.0]))

    def zmp_ref(self, t: float) -> np.ndarray:
        k = int(math.floor(t / self.Ts))
        tau = t - k * self.Ts
        cur = self._contact_center(k, tau / self.Tst)
        if tau < self.Tds:
            prev = self._contact_center(k - 1, (self.Ts + tau) / self.Tst)
            w = tau / self.Tds
            return (1.0 - w) * prev + w * cur
        return cur

    def contact_patch(self, side: str, t: float, corners_local: np.ndarray) -> np.ndarray | None:
        """World xy polygon of a foot's ground contact, or None in swing.

        A flat foot contributes its sole rectangle; a rolled foot only a strip along the
        planted edge, which is all it can push on.
        """
        where = self._where(side, t)
        if where[0] != "stance":
            return None
        f = self.footstep(where[1])
        pitch = self.stance_pitch(where[2])
        c = corners_local.copy()
        if pitch > ROLL_EPS:
            c[:, 0] = np.where(c[:, 0] > 0, self.sole_front, self.sole_front - PIVOT_STRIP)
        elif pitch < -ROLL_EPS:
            c[:, 0] = np.where(c[:, 0] > 0, -self.sole_back + PIVOT_STRIP, -self.sole_back)
        return f.pos + c @ rotz(f.yaw).T

    def support_polygon(self, t: float, corners_local: dict[str, np.ndarray]) -> np.ndarray:
        """Convex support polygon (M,2) in world xy at time t."""
        pts = [p for side in ("left", "right")
               if (p := self.contact_patch(side, t, corners_local[side])) is not None]
        return convex_hull(np.concatenate(pts))


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull, counter-clockwise."""
    pts = sorted(map(tuple, points))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def signed_distance_to_polygon(p: np.ndarray, poly: np.ndarray) -> float:
    """Positive inside a CCW convex polygon, negative outside (metres)."""
    n = len(poly)
    d_min = math.inf
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        L = float(np.hypot(e[0], e[1]))
        if L < 1e-12:
            continue
        # inward normal for CCW polygon is the left normal
        nrm = np.array([-e[1], e[0]]) / L
        d_min = min(d_min, float(np.dot(p - a, nrm)))
    return d_min
