"""STS3215 servo as trained against: numpy port of actuator/bam_kernel.py (BAM, m1-m6).

The training env drives each joint through firmware P -> PWM duty -> DC motor with back-EMF ->
load-dependent friction. gait_studio judges gaits against the same model. Parity with the torch
kernel is checked in tests/test_actuator.py.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np

ERROR_GAIN = 0.166   # firmware: kp * position error [rad] -> PWM duty (BAM, scope-measured)
MAX_PWM = 0.97       # largest duty the firmware outputs

# model -> (load_dependent, directional, stribeck, quadratic)
_MODEL_FLAGS = {
    "m1": (False, False, False, False),
    "m2": (False, False, True, False),
    "m3": (True, False, False, False),
    "m4": (True, False, True, False),
    "m5": (True, True, True, False),
    "m6": (True, True, True, True),
}


@dataclass(frozen=True)
class BamParams:
    model: str
    kt: float
    R: float
    error_gain_ratio: float
    armature: float
    max_velocity: float
    command_delay: float
    friction_base: float
    friction_viscous: float
    friction_stribeck: float = 0.0
    dtheta_stribeck: float = 1.0
    alpha: float = 1.0
    load_friction_base: float = 0.0
    load_friction_stribeck: float = 0.0
    load_friction_motor: float = 0.0
    load_friction_external: float = 0.0
    load_friction_motor_stribeck: float = 0.0
    load_friction_external_stribeck: float = 0.0
    load_friction_motor_quad: float = 0.0
    load_friction_external_quad: float = 0.0
    q_offset: float = 0.0
    actuator: str = ""
    source: str = ""

    @property
    def flags(self) -> tuple[bool, bool, bool, bool]:
        return _MODEL_FLAGS[self.model]

    @property
    def stall_torque(self) -> float:
        """Duty-limited stall torque at 12 V [N-m]."""
        return MAX_PWM * 12.0 * self.kt / self.R

    @classmethod
    def from_json(cls, path: str | Path) -> "BamParams":
        path = Path(path)
        data = json.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"{path}: unknown BAM parameter(s) {sorted(unknown)}")
        model = data.get("model")
        if model not in _MODEL_FLAGS:
            raise ValueError(f"{path}: 'model' must be one of {sorted(_MODEL_FLAGS)}, got {model!r}")
        load_dep, directional, stribeck, quadratic = _MODEL_FLAGS[model]
        required = {"kt", "R", "error_gain_ratio", "armature", "max_velocity",
                    "command_delay", "friction_base", "friction_viscous"}
        if stribeck:
            required |= {"friction_stribeck", "dtheta_stribeck", "alpha"}
        if load_dep and not directional:
            required |= {"load_friction_base"} | ({"load_friction_stribeck"} if stribeck else set())
        if load_dep and directional:
            required |= {"load_friction_motor", "load_friction_external"}
            if stribeck:
                required |= {"load_friction_motor_stribeck", "load_friction_external_stribeck"}
        if quadratic:
            required |= {"load_friction_motor_quad", "load_friction_external_quad"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"{path}: model {model} needs {sorted(missing)}")
        return cls(**data, source=str(path))


def apply_stall_cap(params: BamParams, stall_torque_cap_nm: float | None) -> BamParams:
    """Cap the 12 V stall torque by raising R (no-load speed V/kt is unchanged)."""
    if stall_torque_cap_nm is None or stall_torque_cap_nm >= params.stall_torque:
        return params
    return replace(params, R=params.R * params.stall_torque / float(stall_torque_cap_nm))


class BamActuator:
    """One servo model shared by all joints (the training env has no per-joint differences)."""

    def __init__(self, params: BamParams, kp_fw: float = 32.0, vin: float = 12.0,
                 stall_torque_cap_nm: float | None = None, armature: float | None = None):
        self.p = apply_stall_cap(params, stall_torque_cap_nm)
        self.kp_fw = float(kp_fw)
        self.vin = float(vin)
        self.armature = float(params.armature if armature is None else armature)
        self.friction_scale = 1.0
        self._ring = None

    @classmethod
    def from_config(cls, cfg: dict, base_dir: str | Path) -> "BamActuator":
        path = Path(cfg["params_file"])
        if not path.is_absolute():
            path = Path(base_dir) / path
        return cls(BamParams.from_json(path), kp_fw=cfg.get("kp_fw", 32.0), vin=cfg.get("vin", 12.0),
                   stall_torque_cap_nm=cfg.get("stall_torque_cap_nm"), armature=cfg.get("armature"))

    # ------------------------------------------------------------------ envelope
    @property
    def stall_torque(self) -> float:
        """Duty-limited stall torque at the nominal supply [N-m]."""
        return MAX_PWM * self.vin * self.p.kt / self.p.R

    @property
    def no_load_speed(self) -> float:
        """Joint speed where the back-EMF cancels the supply, at full duty [rad/s]."""
        return MAX_PWM * self.vin / self.p.kt

    @property
    def slew_limit(self) -> float:
        """Firmware limit on how fast the internal target moves [rad/s]."""
        return self.p.max_velocity

    def torque_limit(self, torque: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Largest motor torque magnitude available in the direction of `torque` at speed dq.

        Driving (torque along dq) loses kt^2/R per rad/s to back-EMF; braking gains it.
        """
        k_e = self.p.kt ** 2 / self.p.R
        return np.maximum(self.stall_torque - np.sign(torque) * k_e * dq, 0.0)

    # ------------------------------------------------------------------ friction
    def friction_budget(self, motor_torque, external_torque, dtheta):
        """bam_kernel.BamKernel.friction_budget. Returns (frictionloss, viscous)."""
        p = self.p
        load_dependent, directional, stribeck, quadratic = p.flags
        motor_torque = np.asarray(motor_torque, dtype=float)
        external_torque = np.asarray(external_torque, dtype=float)
        dtheta = np.asarray(dtheta, dtype=float)

        if directional:
            gearbox = np.abs(external_torque * p.load_friction_external - motor_torque * p.load_friction_motor)
            if stribeck:
                gearbox_s = np.abs(external_torque * p.load_friction_external_stribeck
                                   - motor_torque * p.load_friction_motor_stribeck)
        else:
            gearbox = np.abs(external_torque - motor_torque)

        if stribeck:
            coeff = np.exp(-(np.abs(dtheta / p.dtheta_stribeck) ** p.alpha))

        loss = np.full_like(dtheta, p.friction_base)
        if load_dependent:
            loss = loss + (gearbox if directional else p.load_friction_base * gearbox)
        if stribeck:
            loss = loss + coeff * p.friction_stribeck
            if load_dependent:
                if directional:
                    loss = loss + gearbox_s * coeff
                else:
                    loss = loss + p.load_friction_stribeck * gearbox * coeff
                if quadratic:
                    enable = np.sign(external_torque) != np.sign(motor_torque)
                    dir_motor = np.abs(external_torque) < np.abs(motor_torque)
                    dir_ext = np.abs(external_torque) > np.abs(motor_torque)
                    q_motor = p.load_friction_external_quad * np.abs(external_torque) ** 2
                    q_ext = p.load_friction_motor_quad * np.abs(motor_torque) ** 2
                    loss = loss + coeff * (dir_motor * q_motor + dir_ext * q_ext) * enable
        return loss, p.friction_viscous

    def required_motor_torque(self, tau: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Motor torque needed to deliver joint torque `tau` at speed `dq` (friction included).

        Quasi-static: the external load is taken as -tau and friction fully opposes motion above
        ~0.02 rad/s. At standstill stiction is ignored, so the estimate is conservative there.
        """
        direction = np.clip(dq / 0.02, -1.0, 1.0)
        ext = -tau
        motor = tau
        for _ in range(2):
            loss, damping = self.friction_budget(motor, ext, dq)
            budget = (loss + damping * np.abs(dq)) * self.friction_scale
            motor = tau + direction * budget
        return motor

    # ------------------------------------------------------------------ simulation
    def reset(self):
        self._ring = None

    def motor(self, q_goal: np.ndarray, q: np.ndarray, dq: np.ndarray, dt: float) -> np.ndarray:
        """Firmware + DC motor: goal and encoder state -> motor torque [N-m]. Stateful."""
        p = self.p
        if self._ring is None:
            d = p.command_delay / dt
            self._d0 = int(math.floor(d))
            self._frac = d - self._d0
            self._ring = np.tile(q_goal, (self._d0 + 2, 1))
            self._head = 0
            self._q_smooth = np.array(q, dtype=float)
        n = len(self._ring)
        self._head = (self._head + 1) % n
        self._ring[self._head] = q_goal
        i0 = (self._head - self._d0) % n
        i1 = (self._head - self._d0 - 1) % n
        goal = (1.0 - self._frac) * self._ring[i0] + self._frac * self._ring[i1]

        step = p.max_velocity * dt
        self._q_smooth = np.clip(goal, self._q_smooth - step, self._q_smooth + step)
        duty = np.clip((self._q_smooth - q) * self.kp_fw * ERROR_GAIN * p.error_gain_ratio, -MAX_PWM, MAX_PWM)
        return p.kt * self.vin * duty / p.R - (p.kt ** 2) * dq / p.R

    def stop_clip(self, motor_torque, external_torque, dq, inertia_eff, dt: float) -> np.ndarray:
        """Friction torque to apply; can stop a joint within one step but never reverse it."""
        loss, damping = self.friction_budget(motor_torque, external_torque, dq)
        budget = (loss + damping * np.abs(dq)) * self.friction_scale
        tau_stop = (inertia_eff / dt) * dq + (motor_torque + external_torque)
        return -np.sign(tau_stop) * np.minimum(np.abs(tau_stop), budget)
