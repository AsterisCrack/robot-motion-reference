"""Open-loop physics preview: the free-floating robot tracks the reference joint angles through the
same BAM servo model as the training sim (gait_studio/actuator.py). No balance feedback, so it will
drift and often fall; the point is to see whether a gait is roughly dynamically plausible before
spending a training run.
"""
from __future__ import annotations

import copy
import math

import mujoco
import numpy as np

from .robot import RobotModel


class PhysicsPreview:
    def __init__(self, robot: RobotModel):
        self.r = robot
        self.act = copy.copy(robot.actuator)   # the servo is stateful; don't share it
        self.m = mujoco.MjModel.from_xml_string(robot.xml)
        self.m.dof_armature[robot.dof_adr] = float(robot.armature)
        # The BAM servo supplies all joint friction, and its back-EMF braking can exceed the
        # MJCF's +-stall ctrlrange. Drop the MJCF's own friction and torque clamp.
        self.m.dof_frictionloss[robot.dof_adr] = 0.0
        self.m.dof_damping[robot.dof_adr] = 0.0
        self.m.actuator_ctrllimited[:] = 0
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
        self.act.reset()

    def _torque(self, target: np.ndarray) -> np.ndarray:
        r, d, m = self.r, self.d, self.m
        q = d.qpos[r.qpos_adr]
        qd = d.qvel[r.dof_adr]
        tau_motor = self.act.motor(target, q, qd, self.sim_dt)
        # Load on the joint from everything but the servo (gravity, contact, coupling); one step
        # old, as it comes from the previous forward pass.
        ext = (d.qfrc_constraint + d.qfrc_passive - d.qfrc_bias)[r.dof_adr]
        inertia = d.qM[m.dof_Madr[r.dof_adr]]
        return tau_motor + self.act.stop_clip(tau_motor, ext, qd, inertia, self.sim_dt)

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
