"""Speed-dependent cadence: how long one gait cycle lasts for a command.

Humans split a speed change between step length and step frequency. Step length grows as a power
of speed, s ∝ v^β with β ≈ 0.42 (Grieve 1968, explained by Kuo 2001) to 0.58 (Wu et al. 2019, very
slow walking), so cadence grows as v^(1-β). In leg-length units the law holds across body sizes
(dynamic similarity, Alexander; Hof 1996):

    s / L = 1.15 · v'^β,   v' = v / sqrt(g L)      =>   f_step · sqrt(L/g) = v'^(1-β) / 1.15

This robot (L ≈ 0.21 m) tops out at v' ≈ 0.14 (Froude ≈ 0.02), the human very-slow-walking regime.
Sideways walking is scored at 1/0.45 of its speed (preferred sideways speed is ~0.45 of forward,
Handford & Srinivasan 2014). Turning keeps cadence roughly constant and changes stride length
(Courtine & Schieppati 2003); it enters as the speed a foot sweeps at half the stance width.

The law is anchored at the robot's top forward command: there the cadence is the human one for
any exponent, so the exponent only shapes how fast the cadence drops toward slower commands.
"""
from __future__ import annotations

import math

G = 9.81
STEP_LENGTH_COEFF = 1.15        # s/L = 1.15 v'^beta (Hof 1996)
HUMAN_EXPONENT = 0.5            # 1 - beta for the reference level (Hof 1996 / walk ratio)
SIDEWAYS_SPEED_RATIO = 0.45     # preferred sideways / forward speed (Handford & Srinivasan 2014)


def effective_speed(cmd, stance_width: float) -> float:
    """Forward-walking speed that 'costs' the same cadence as this command."""
    vx, vy, wz = (float(c) for c in cmd)
    return math.sqrt(vx * vx + (vy / SIDEWAYS_SPEED_RATIO) ** 2 + (wz * 0.5 * stance_width) ** 2)


def human_step_frequency(v_eff: float, leg_length: float, v_ref: float, exponent: float) -> float:
    """Step frequency [Hz] from the human law, pivoted at v_ref (see module docstring)."""
    if v_eff <= 1e-9:
        return 0.0 if exponent > 0.0 else human_step_frequency(v_ref, leg_length, v_ref, HUMAN_EXPONENT)
    scale = math.sqrt(G * leg_length)
    v_ref_n = max(v_ref, 1e-6) / scale
    f_ref_n = v_ref_n ** HUMAN_EXPONENT / STEP_LENGTH_COEFF          # human cadence at v_ref
    f_n = f_ref_n * (v_eff / max(v_ref, 1e-6)) ** exponent
    return f_n / math.sqrt(leg_length / G)


def period_for(cmd, style: dict, leg_length: float, v_ref: float) -> float:
    """Gait cycle (two steps) for a command, in seconds. Continuous, not snapped to any grid."""
    v_eff = effective_speed(cmd, style["stance_width"])
    f_step = style["cadence_scale"] * human_step_frequency(v_eff, leg_length, v_ref, style["cadence_exponent"])
    T = 2.0 / f_step if f_step > 1e-9 else math.inf
    T = min(max(T, style["period_min"]), style["period_max"])
    wz = abs(float(cmd[2]))
    if wz > 1e-9:   # a slow cadence while turning means a big yaw per step; cap it
        T = min(T, 2.0 * math.radians(style["max_step_yaw"]) / wz)
    return max(T, style["period_min"])
