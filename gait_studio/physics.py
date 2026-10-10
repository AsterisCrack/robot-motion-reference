"""Open-loop physics preview: the free-floating robot tracks the reference joint angles with the
same PD + torque clamp as the training sim. No balance feedback, so it will drift and often fall;
the point is to see whether a gait is roughly dynamically plausible before spending a training run.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .robot import RobotModel


class PhysicsPreview:
    def __init__(self, robot: RobotModel):
        self.r = robot
        self.m = mujoco.MjModel.from_xml_string(robot.xml)
        if robot.armature is not None:
            self.m.dof_armature[robot.dof_adr] = float(robot.armature)
        self.d = mujoco.MjData(self.m)
        # actuator index per table joint (actuators are named after their joints)
        self.act_idx = np.array([mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                                 for n in robot.joint_names])
        if (self.act_idx < 0).any():
            raise RuntimeError("MJCF actuators must be named after their joints")
        self.sim_dt = float(self.m.opt.timestep)
        self.fallen = False
        self.time = 0.0

    def reset(self, qpos: np.ndarray, lift: float = 0.002):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = qpos
        self.d.qpos[0:2] = 0.0
        mujoco.mj_kinematics(self.m, self.d)
        sole_z = min(self.d.site_xpos[self.r.sole_site_ids[s]][2] for s in ("left", "right"))
        self.d.qpos[2] += lift - sole_z
        mujoco.mj_forward(self.m, self.d)
        self.fallen = False
        self.time = 0.0

    def _torque(self, target: np.ndarray) -> np.ndarray:
        r = self.r
        q = self.d.qpos[r.qpos_adr]
        qd = self.d.qvel[r.dof_adr]
        tau = r.kp * (target - q) - r.kd * qd
        # DCMotor-style envelope: stall * (1 - w / w_max), then the continuous clamp.
        env = r.stall_torque * np.clip(1.0 - np.sign(tau) * qd / r.velocity_limit, 0.0, 1.0)
        return np.clip(tau, -np.minimum(env, r.effort_limit), np.minimum(env, r.effort_limit))

    def step(self, target: np.ndarray, duration: float):
        n = max(1, int(round(duration / self.sim_dt)))
        for _ in range(n):
            self.d.ctrl[self.act_idx] = self._torque(target)
            mujoco.mj_step(self.m, self.d)
        self.time += n * self.sim_dt
        up = self.d.xmat[self.r.base_id].reshape(3, 3)[:, 2]
        if self.d.qpos[2] < 0.12 or math.degrees(math.acos(np.clip(up[2], -1, 1))) > 45.0:
            self.fallen = True

    @property
    def qpos(self) -> np.ndarray:
        return self.d.qpos.copy()
