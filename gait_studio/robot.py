"""Robot model wrapper: loads the training MJCF, adds sole sites, exposes joint bookkeeping."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import mujoco
import numpy as np
import yaml

from .actuator import BamActuator

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROBOT_CONFIG = os.path.join(_HERE, "configs", "robot_v2.yaml")

SIDES = ("left", "right")


@dataclass
class Sole:
    """Sole rectangle in the foot body frame (metres)."""
    center: np.ndarray   # (3,) sole centre on the contact plane, foot body frame
    front: float         # +x extent from centre
    back: float          # -x extent from centre (positive number)
    half_width: float    # half extent along y


class RobotModel:
    def __init__(self, config_path: str = DEFAULT_ROBOT_CONFIG):
        self.config_path = os.path.abspath(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        cfg_dir = os.path.dirname(self.config_path)
        self.mjcf_path = os.path.normpath(os.path.join(cfg_dir, self.cfg["mjcf"]))
        self.mesh_dir = os.path.normpath(os.path.join(cfg_dir, self.cfg["mesh_dir"]))

        xml = self._read_xml()
        # Pass 1: measure the soles from the foot meshes.
        probe = mujoco.MjModel.from_xml_string(xml)
        self.soles = {side: self._measure_sole(probe, self.cfg["foot_bodies"][side]) for side in SIDES}
        # Pass 2: reload with sole sites so IK can target the contact patch directly.
        xml = self._inject_sole_sites(xml)
        self.xml = xml
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)

        self.base_id = self.m.body(self.cfg["base_body"]).id
        self.foot_body_ids = {s: self.m.body(self.cfg["foot_bodies"][s]).id for s in SIDES}
        self.sole_site_ids = {s: self.m.site(f"{s}_sole").id for s in SIDES}

        self.joint_names: list[str] = list(self.cfg["joints"])
        self.num_joints = len(self.joint_names)
        self.qpos_adr = np.array([self.m.joint(n).qposadr[0] for n in self.joint_names])
        self.dof_adr = np.array([self.m.joint(n).dofadr[0] for n in self.joint_names])
        lim = np.radians(np.array(self.cfg["joint_limits_deg"], dtype=float))
        self.lower, self.upper = lim[:, 0], lim[:, 1]

        self.actuator = BamActuator.from_config(self.cfg["actuator"], cfg_dir)
        self.armature = self.actuator.armature
        self.command_ranges = {k: tuple(v) for k, v in self.cfg["command_ranges"].items()}
        self.control_dt = float(self.cfg["control_dt"])
        self.stand_threshold = float(self.cfg.get("stand_threshold", 0.05))

        # Left/right mirror. The V2 model is a mirrored CAD, so the reflection is a pure
        # permutation of joint values (see BipedRobotV2EnvCfg.mirror_joint_signs).
        self.mirror_perm = np.array([self.joint_names.index(self._swap_side(n)) for n in self.joint_names])
        self.mirror_signs = np.ones(self.num_joints)

        self.total_mass = float(self.m.body_subtreemass[self.base_id])
        self.straight_com_height, self.straight_base_height = self._straight_heights()

    # ------------------------------------------------------------------ loading
    def _read_xml(self) -> str:
        with open(self.mjcf_path, "r", encoding="utf-8") as f:
            xml = f.read()
        mesh_dir = self.mesh_dir.replace("\\", "/")
        return re.sub(r'file="[^"]*[/\\]meshes[/\\]', f'file="{mesh_dir}/', xml)

    @staticmethod
    def _measure_sole(m: mujoco.MjModel, body_name: str) -> Sole:
        bid = m.body(body_name).id
        pts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            pts.append(v @ R.reshape(3, 3).T + m.geom_pos[g])
        if not pts:
            raise RuntimeError(f"No mesh geoms on foot body {body_name}")
        p = np.concatenate(pts)
        zmin = p[:, 2].min()
        # Only the points near the bottom define the contact patch.
        bottom = p[p[:, 2] < zmin + 0.003]
        lo, hi = bottom.min(0), bottom.max(0)
        center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, zmin])
        return Sole(center=center, front=hi[0] - center[0], back=center[0] - lo[0],
                    half_width=(hi[1] - lo[1]) / 2)

    def _inject_sole_sites(self, xml: str) -> str:
        for side in SIDES:
            body = self.cfg["foot_bodies"][side]
            c = self.soles[side].center
            site = f'<site name="{side}_sole" pos="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" size="0.004" rgba="1 0.3 0.3 1"/>'
            pattern = rf'(<body name="{re.escape(body)}"[^>]*>)'
            xml, n = re.subn(pattern, rf"\1\n{site}", xml, count=1)
            if n != 1:
                raise RuntimeError(f"Could not find body {body} in MJCF")
        return xml

    @staticmethod
    def _swap_side(name: str) -> str:
        if name.startswith("l_"):
            return "r_" + name[2:]
        if name.startswith("r_"):
            return "l_" + name[2:]
        raise ValueError(f"Joint name without l_/r_ prefix: {name}")

    def _straight_heights(self) -> tuple[float, float]:
        q = self.make_qpos(np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(self.num_joints))
        self.forward(q)
        sole_z = min(self.d.site_xpos[self.sole_site_ids[s]][2] for s in SIDES)
        return float(self.d.subtree_com[self.base_id][2] - sole_z), float(-sole_z)

    # ------------------------------------------------------------------ helpers
    def make_qpos(self, base_pos, base_quat, joints) -> np.ndarray:
        q = np.zeros(self.m.nq)
        q[0:3] = base_pos
        q[3:7] = base_quat
        q[self.qpos_adr] = joints
        return q

    def joints_of(self, qpos: np.ndarray) -> np.ndarray:
        return qpos[..., self.qpos_adr]

    def forward(self, qpos: np.ndarray) -> mujoco.MjData:
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d

    def sole_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        sid = self.sole_site_ids[side]
        return self.d.site_xpos[sid].copy(), self.d.site_xmat[sid].reshape(3, 3).copy()

    def mirror_joints(self, q: np.ndarray) -> np.ndarray:
        return q[..., self.mirror_perm] * self.mirror_signs

    def sole_corners_local(self, side: str) -> np.ndarray:
        """Sole rectangle corners (4,2) relative to the sole centre, in the foot frame xy."""
        s = self.soles[side]
        return np.array([[s.front, s.half_width], [s.front, -s.half_width],
                         [-s.back, -s.half_width], [-s.back, s.half_width]])

    def summary(self) -> str:
        sl = self.soles["left"]
        return (f"mass {self.total_mass:.3f} kg | straight-leg CoM height {self.straight_com_height:.3f} m"
                f" | sole {sl.front + sl.back:.3f} x {2 * sl.half_width:.3f} m,"
                f" centre offset (x {sl.center[0]:+.4f}, y {sl.center[1]:+.4f}, z {sl.center[2]:+.4f})")


if __name__ == "__main__":
    print(RobotModel().summary())
