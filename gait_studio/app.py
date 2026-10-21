"""Gait Studio: tune a procedural walking style live in the browser, check it against the servo,
then bake the gait table for training.

    python -m gait_studio.app            # then open http://localhost:8080
"""
from __future__ import annotations

import argparse
import glob
import math
import os

# One BLAS thread: the IK solves thousands of tiny systems per cycle, and multithreaded OpenBLAS
# overhead makes that several times slower (badly so on Windows). Same as bake.py.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import threading
import time
import traceback

import mujoco
import numpy as np
import viser
import viser.uplot as uplot

from .analysis import FAIL, OK, WARN, Analyzer
from .bake import bake
from .gait import FIELD_SLICES, GaitGenerator
from .physics import PhysicsPreview
from .robot import DEFAULT_ROBOT_CONFIG, SIDES, RobotModel
from .style import ANCHORS, GROUPS, KNOB_BY_NAME, KNOBS, Style

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE_DIR = os.path.join(HERE, "styles")
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "processed", "gait_table.npz"))

BASE_LABEL = "Base (step in place)"
ICON = {OK: "🟢", WARN: "🟡", FAIL: "🔴"}
SIDE_COLOR = {"left": (60, 140, 255), "right": (255, 110, 60)}
PRESETS = {
    "Stop": (0.0, 0.0, 0.0), "Fwd": (1.0, 0.0, 0.0), "Back": (-1.0, 0.0, 0.0),
    "Left": (0.0, 1.0, 0.0), "Right": (0.0, -1.0, 0.0), "Turn L": (0.0, 0.0, 1.0), "Turn R": (0.0, 0.0, -1.0),
}


# --------------------------------------------------------------------------- planar transforms
def _planar(x=0.0, y=0.0, yaw=0.0):
    return np.array([x, y, yaw], dtype=float)


def _compose(a, b):
    """a ∘ b for planar poses (x, y, yaw)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]])


def _inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([-(c * a[0] + s * a[1]), -(-s * a[0] + c * a[1]), -a[2]])


def _yaw_of_quat(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _base_planar(qpos):
    return _planar(qpos[0], qpos[1], _yaw_of_quat(qpos[3:7]))


def _apply_xy(T, pts):
    pts = np.asarray(pts, dtype=float)
    c, s = math.cos(T[2]), math.sin(T[2])
    out = pts.copy()
    out[..., 0] = T[0] + c * pts[..., 0] - s * pts[..., 1]
    out[..., 1] = T[1] + s * pts[..., 0] + c * pts[..., 1]
    return out


def _quat_z(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


# --------------------------------------------------------------------------- robot meshes
class RobotMeshes:
    """The robot's mesh geoms as viser nodes, posed from a qpos through MuJoCo FK."""

    def __init__(self, server: viser.ViserServer, robot: RobotModel, name: str, color, opacity=None):
        self.m = robot.m
        self.d = mujoco.MjData(robot.m)
        self.handles = []
        for g in range(self.m.ngeom):
            if self.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = self.m.geom_dataid[g]
            v = self.m.mesh_vert[self.m.mesh_vertadr[mid]: self.m.mesh_vertadr[mid] + self.m.mesh_vertnum[mid]]
            f = self.m.mesh_face[self.m.mesh_faceadr[mid]: self.m.mesh_faceadr[mid] + self.m.mesh_facenum[mid]]
            h = server.scene.add_mesh_simple(f"/{name}/geom{g}", v.astype(np.float32), f.astype(np.uint32),
                                             color=color, opacity=opacity, flat_shading=False)
            self.handles.append((g, h))

    def update(self, qpos: np.ndarray, T=None):
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        q = np.zeros(4)
        qz = _quat_z(T[2]) if T is not None else None
        for g, h in self.handles:
            mujoco.mju_mat2Quat(q, self.d.geom_xmat[g])
            p = self.d.geom_xpos[g].copy()
            if T is not None:
                p[:2] = _apply_xy(T, p[:2])
                out = np.zeros(4)
                mujoco.mju_mulQuat(out, qz, q)
                h.wxyz, h.position = out, p
            else:
                h.wxyz, h.position = q.copy(), p

    def set_visible(self, visible: bool):
        for _, h in self.handles:
            h.visible = visible


# --------------------------------------------------------------------------- the app
class Studio:
    def __init__(self, robot_cfg: str, style_path: str | None, port: int, host: str = "127.0.0.1"):
        self.robot = RobotModel(robot_cfg)
        self.gen = GaitGenerator(self.robot)
        self.analyzer = Analyzer(self.robot)
        self.physics = PhysicsPreview(self.robot)
        self.style_path = style_path or os.path.join(STYLE_DIR, "default.yaml")
        self.style = Style.load(self.style_path) if os.path.exists(self.style_path) else Style()
        self.ranges = self.robot.command_ranges
        self.cmd = np.zeros(3)

        self.lock = threading.Lock()
        self.dirty = threading.Event()
        self.cycle = None
        self.report = None
        self.stand_q = None
        self.gen_ms = 0.0
        self._syncing = False

        # playback state
        self.play_t = 0.0
        self.offset = _planar()
        self.loop_disp = _planar()
        self.last_frame = 0
        self.clip = None
        self.clip_t = 0.0
        self.physics_reset_at = None
        # pose blend across regenerations (joints + base height), so switching gaits doesn't snap
        self.shown_q = None
        self.blend_from = None
        self.blend_t0 = 0.0
        self.blend_time = 0.25

        self.server = viser.ViserServer(host=host, port=port, label="Gait Studio")
        self.server.gui.configure_theme(control_width="large", dark_mode=False)
        self._build_scene()
        self._build_gui()
        self.dirty.set()

        threading.Thread(target=self._regen_loop, daemon=True).start()
        threading.Thread(target=self._render_loop, daemon=True).start()

    # ------------------------------------------------------------------ scene
    def _build_scene(self):
        sc = self.server.scene
        sc.add_grid("/grid", width=3.0, height=3.0, cell_size=0.05, section_size=0.25, plane="xy")
        sc.set_up_direction("+z")
        self.server.initial_camera.position = (0.55, -0.75, 0.42)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.12)
        # Everything that belongs to the reference gait hangs under /disp and is written in
        # cycle-local coordinates; the /disp pose alone decides whether the robot walks across
        # the floor (follow) or stays in frame with the floor sliding under it (walk in place).
        self.disp = sc.add_frame("/disp", show_axes=False)
        self.robot_viz = RobotMeshes(self.server, self.robot, "disp/robot", (200, 205, 215))
        self.phys_viz = RobotMeshes(self.server, self.robot, "physics", (255, 170, 60), opacity=0.9)
        self.phys_viz.set_visible(False)
        self.clip_viz = RobotMeshes(self.server, self.robot, "clip", (150, 120, 220), opacity=0.55)
        self.clip_viz.set_visible(False)
        self.com_h = sc.add_icosphere("/disp/overlay/com", radius=0.008, color=(40, 200, 90))
        self.com_ground_h = sc.add_icosphere("/disp/overlay/com_ground", radius=0.005, color=(40, 200, 90))
        self.zmp_h = sc.add_icosphere("/disp/overlay/zmp", radius=0.006, color=(230, 40, 40))
        self.support_h = None
        self.trail_h = {}
        self.step_h = []
        self.phys_label = sc.add_label("/physics_label", "physics", position=(0, -0.35, 0.33), visible=False)

    def _draw_static_overlays(self):
        """Footsteps and swing trails of the current cycle (cycle-local, under /disp)."""
        sc = self.server.scene
        cyc = self.cycle
        for h in self.step_h:
            h.remove()
        self.step_h = []
        for h in self.trail_h.values():
            h.remove()
        self.trail_h = {}
        if cyc is None:
            return
        show_steps = self.gui_steps.value
        corners = {s: self.robot.sole_corners_local(s) for s in SIDES}
        for n, (side, xy, yaw) in enumerate(cyc.footsteps):
            c, s = math.cos(yaw), math.sin(yaw)
            rect = xy + corners[side] @ np.array([[c, -s], [s, c]]).T
            pts = np.concatenate([rect, rect[:1]])
            seg = np.stack([pts[:-1], pts[1:]], axis=1)
            seg3 = np.concatenate([seg, np.full(seg.shape[:2] + (1,), 0.001)], axis=-1)
            self.step_h.append(sc.add_line_segments(f"/disp/overlay/step{n}", seg3, SIDE_COLOR[side], thickness=3.0,
                                                    thickness_units="screen", visible=show_steps))
        if self.gui_trails.value:
            for side in SIDES:
                p = cyc.sole_pos[side]
                if np.ptp(p, axis=0).max() < 1e-4:   # standing: nothing to trace
                    continue
                self.trail_h[side] = sc.add_spline_catmull_rom(f"/disp/overlay/trail_{side}", p, color=SIDE_COLOR[side],
                                                               thickness=2.0, thickness_units="screen")

    # ------------------------------------------------------------------ gui
    def _build_gui(self):
        gui = self.server.gui
        self.status_md = gui.add_markdown("**Gait Studio** - generating…")
        tabs = gui.add_tab_group()

        # ---------------- Drive
        with tabs.add_tab("Drive", icon=viser.Icon.STEERING_WHEEL):
            self.sl_cmd = []
            for i, (k, label, unit) in enumerate([("vx", "Forward vx", "m/s"), ("vy", "Lateral vy", "m/s"),
                                                  ("wz", "Turn wz", "rad/s")]):
                lo, hi = self.ranges[k]
                sl = gui.add_slider(f"{label} [{unit}]", lo, hi, 0.01, 0.0)
                sl.on_update(self._make_cmd_cb(i))
                self.sl_cmd.append(sl)
            presets = gui.add_button_group("Preset", list(PRESETS))
            presets.on_click(lambda e: self._set_preset(e.target.value))
            self.gui_tour = gui.add_checkbox("Auto tour (cycle presets)", False)
            self.gui_inplace = gui.add_checkbox("Walk in place (robot stays in frame)", False,
                                                hint="The floor, footsteps and trails slide under the robot instead.")
            self.gui_step_still = gui.add_checkbox("Step in place when stopped", False,
                                                   hint="At zero command the robot stands still (as in training). "
                                                        "Tick to preview the Base gait instead.")
            self.gui_step_still.on_update(lambda _: self.dirty.set())
            gui.add_button("Reset drive", icon=viser.Icon.REFRESH).on_click(lambda _: self._reset_drive())
            gui.add_divider()
            self.gui_speed = gui.add_slider("Playback speed", 0.05, 2.0, 0.05, 1.0)
            self.gui_pause = gui.add_checkbox("Pause", False)
            self.gui_phase = gui.add_slider("Phase (when paused)", 0.0, 1.0, 0.01, 0.0)
            gui.add_divider()
            self.gui_steps = gui.add_checkbox("Show footsteps", True)
            self.gui_trails = gui.add_checkbox("Show swing trails", True)
            self.gui_balance = gui.add_checkbox("Show CoM (green) / ZMP (red) / support", True)
            for h in (self.gui_steps, self.gui_trails):
                h.on_update(lambda _: self._draw_static_overlays())
            gui.add_divider()
            self.gui_physics = gui.add_checkbox("Physics preview (open-loop BAM servo)", False,
                                                hint="Orange robot: MuJoCo sim tracking the reference joints through the "
                                                     "training BAM servo model, no balance feedback.")
            self.gui_physics.on_update(lambda e: self._toggle_physics())
            gui.add_button("Reset physics").on_click(lambda _: self._reset_physics())
            gui.add_divider()
            self.gui_clip_path = gui.add_text("Clip ghost NPZ", os.path.join("processed", "StandardWalk.npz"),
                                              hint="Joint-angle NPZ from fbx_to_joint_data.py, shown in purple next to the robot.")
            self.gui_clip_speed = gui.add_slider("Clip speed", 0.25, 3.0, 0.05, 1.9,
                                                 hint="Froude scaling human→this robot is about 1.9x.")
            self.gui_clip = gui.add_checkbox("Show clip ghost", False)
            self.gui_clip.on_update(lambda _: self._toggle_clip())

        # ---------------- Style
        with tabs.add_tab("Style", icon=viser.Icon.ADJUSTMENTS):
            self.gui_anchor = gui.add_dropdown("Editing", [BASE_LABEL] + [a.label for a in ANCHORS],
                                               initial_value=BASE_LABEL,
                                               hint="Anchors override knobs at the extremes of the command space; "
                                                    "values blend linearly toward the base.")
            self.gui_follow = gui.add_checkbox("Drive to the anchor command", True)
            self.gui_anchor.on_update(lambda _: self._on_anchor_change())
            self.override_md = gui.add_markdown("")
            gui.add_button("Clear overrides on this anchor").on_click(lambda _: self._clear_overrides())
            gui.add_button("Reset all knobs", icon=viser.Icon.RESTORE, color="red").on_click(self._confirm_reset_all)
            self.knob_sl = {}
            self.knob_reset = {}
            for group in GROUPS:
                with gui.add_folder(group):
                    for k in KNOBS:
                        if k.group != group:
                            continue
                        label = f"{k.label} [{k.unit}]" + ("" if k.anchorable else " ·global")
                        sl = gui.add_slider(label, k.min, k.max, k.step, self.style.get(k.name), hint=k.help or None)
                        sl.on_update(self._make_knob_cb(k.name))
                        self.knob_sl[k.name] = sl
                        # Only shown while this knob differs from a fresh reset.
                        btn = gui.add_button(f"↺ reset {k.label}", icon=viser.Icon.ARROW_BACK_UP, color="gray",
                                             visible=False)
                        btn.on_click(self._make_knob_reset_cb(k.name))
                        self.knob_reset[k.name] = btn
            self._sync_knobs()

        # ---------------- Check
        with tabs.add_tab("Check", icon=viser.Icon.HEART_RATE_MONITOR):
            self.check_md = gui.add_markdown("")
            self.gui_joint = gui.add_dropdown("Joint", self.robot.joint_names, initial_value=self.robot.joint_names[6])
            self.gui_joint.on_update(lambda _: self._update_plots())
            x = np.linspace(0, 1, 10)
            z = np.zeros(10)
            common = dict(aspect=1.6, legend=uplot.Legend(show=True), scales={"x": uplot.Scale(time=False)})
            self.plot_angle = gui.add_uplot((x, z, z, z), (uplot.Series(label="phase"),
                                            uplot.Series(label="angle °", stroke="#2563eb", width=2),
                                            uplot.Series(label="lower", stroke="#ef4444", dash=(4, 4)),
                                            uplot.Series(label="upper", stroke="#ef4444", dash=(4, 4))),
                                            title="Joint angle", **common)
            self.plot_speed = gui.add_uplot((x, z, z), (uplot.Series(label="phase"),
                                            uplot.Series(label="|speed| rad/s", stroke="#7c3aed", width=2),
                                            uplot.Series(label="servo limit", stroke="#ef4444", dash=(4, 4))),
                                            title="Joint speed", **common)
            self.plot_usage = gui.add_uplot((x, z, z, z), (uplot.Series(label="phase"),
                                            uplot.Series(label="this joint %", stroke="#ea580c", width=2),
                                            uplot.Series(label="worst joint %", stroke="#a3a3a3", width=1),
                                            uplot.Series(label="100%", stroke="#ef4444", dash=(4, 4))),
                                            title="Motor torque vs BAM torque-speed envelope", **common)
            self.plot_feet = gui.add_uplot((x, z, z, z), (uplot.Series(label="phase"),
                                           uplot.Series(label="left sole mm", stroke="#3b82f6", width=2),
                                           uplot.Series(label="right sole mm", stroke="#f97316", width=2),
                                           uplot.Series(label="ZMP margin mm", stroke="#16a34a", width=2)),
                                           title="Foot height & balance margin", **common)

        # ---------------- Bake & files
        with tabs.add_tab("Bake", icon=viser.Icon.DEVICE_FLOPPY):
            self.gui_name = gui.add_text("Style name", self.style.name)
            self.gui_style_file = gui.add_dropdown("Preset file", self._style_files(),
                                                   initial_value=os.path.basename(self.style_path)
                                                   if os.path.basename(self.style_path) in self._style_files() else None)
            with gui.add_folder("Presets"):
                gui.add_button("Load preset").on_click(lambda _: self._load_style())
                gui.add_button("Save preset (as name.yaml)").on_click(lambda _: self._save_style())
            gui.add_divider()
            self.gui_out = gui.add_text("Table output", DEFAULT_OUT)
            self.gui_bake = gui.add_button("Bake gait table", color="green", icon=viser.Icon.PLAYER_PLAY)
            self.gui_bake.on_click(lambda _: threading.Thread(target=self._bake, daemon=True).start())
            self.bake_bar = gui.add_progress_bar(0.0, visible=False)
            self.bake_md = gui.add_markdown("Bakes one cycle per grid command (see configs/robot_v2.yaml `grid`) "
                                            "and writes the table, a style copy and a feasibility report.")

    # ------------------------------------------------------------------ callbacks
    def _make_cmd_cb(self, i):
        def cb(_):
            if self._syncing:
                return
            self.cmd[i] = self.sl_cmd[i].value
            self.dirty.set()
        return cb

    def _set_cmd(self, cmd):
        self._syncing = True
        for i, v in enumerate(cmd):
            self.sl_cmd[i].value = float(v)
            self.cmd[i] = float(v)
        self._syncing = False
        self.dirty.set()

    def _set_preset(self, name):
        u = PRESETS[name]
        cmd = []
        for v, k in zip(u, ("vx", "vy", "wz")):
            lo, hi = self.ranges[k]
            cmd.append(0.0 if v == 0 else (0.6 * hi if v > 0 else 0.6 * lo))
        self._set_cmd(cmd)

    def _current_anchor(self):
        label = self.gui_anchor.value
        for a in ANCHORS:
            if a.label == label:
                return a.name
        return None

    def _make_knob_cb(self, name):
        def cb(_):
            if self._syncing:
                return
            with self.lock:
                self.style.set(name, float(self.knob_sl[name].value), self._current_anchor())
            self._sync_knobs()
            self.dirty.set()
        return cb

    def _make_knob_reset_cb(self, name):
        def cb(_):
            with self.lock:
                self.style.reset_knob(name, self._current_anchor())
            self._sync_knobs()
            self.dirty.set()
        return cb

    def _confirm_reset_all(self, event):
        client = event.client
        if client is None:
            self._reset_style()
            return
        with client.gui.add_modal("Reset all style knobs?") as modal:
            client.gui.add_markdown("Base and every anchor go back to the defaults. The style name is kept; "
                                    "saved preset files are not touched.")
            ok = client.gui.add_button("Reset everything", color="red")
            cancel = client.gui.add_button("Cancel")

            @ok.on_click
            def _(_):
                self._reset_style()
                modal.close()

            @cancel.on_click
            def _(_):
                modal.close()

    def _reset_drive(self):
        self._syncing = True
        self.gui_speed.value = 1.0
        self.gui_pause.value = False
        self.gui_phase.value = 0.0
        self.gui_tour.value = False
        self._syncing = False
        with self.lock:
            self.offset = _planar()
            self.play_t = 0.0
        self._set_cmd((0.0, 0.0, 0.0))

    def _sync_knobs(self):
        anchor = self._current_anchor()
        self._syncing = True
        for k in KNOBS:
            sl = self.knob_sl[k.name]
            at = anchor if k.anchorable else None
            sl.value = float(self.style.get(k.name, at))
            sl.disabled = bool(k.signed and anchor is None)
            self.knob_reset[k.name].visible = self.style.is_modified(k.name, at)
        self._syncing = False
        if anchor is None:
            signed = ", ".join(k.label for k in KNOBS if k.signed)
            self.override_md.content = (f"Editing the **base** gait (zero command). Leans ({signed}) are set on the "
                                        "Sideways/Turning anchors.")
        else:
            ov = self.style.anchors[anchor]
            names = ", ".join(f"{KNOB_BY_NAME[k].label} = {v:g}" for k, v in ov.items()) or "none yet"
            self.override_md.content = f"Overrides on **{anchor}**: {names}"

    def _on_anchor_change(self):
        anchor = self._current_anchor()
        self._sync_knobs()
        if self.gui_follow.value:
            a = next((x for x in ANCHORS if x.name == anchor), None)
            if a is None:
                self._set_cmd((0.0, 0.0, 0.0))
            else:
                cmd = [0.0, 0.0, 0.0]
                i = ("vx", "vy", "wz").index(a.axis)
                lo, hi = self.ranges[a.axis]
                cmd[i] = hi if a.direction > 0 else lo
                self._set_cmd(cmd)

    def _clear_overrides(self):
        anchor = self._current_anchor()
        if anchor is not None:
            with self.lock:
                self.style.anchors[anchor] = {}
            self._sync_knobs()
            self.dirty.set()

    def _style_files(self):
        files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(STYLE_DIR, "*.yaml")))
        return files or ["default.yaml"]

    def _load_style(self):
        fname = self.gui_style_file.value
        path = os.path.join(STYLE_DIR, fname)
        if os.path.exists(path):
            with self.lock:
                self.style = Style.load(path)
            self.gui_name.value = self.style.name
            self._sync_knobs()
            self.dirty.set()

    def _save_style(self):
        name = "".join(c for c in self.gui_name.value.strip() if c.isalnum() or c in "-_") or "style"
        with self.lock:
            self.style.name = name
            path = os.path.join(STYLE_DIR, f"{name}.yaml")
            self.style.save(path)
        self.gui_style_file.options = self._style_files()
        self.gui_style_file.value = f"{name}.yaml"
        self.bake_md.content = f"Saved `{path}`"

    def _reset_style(self):
        with self.lock:
            self.style = Style(name=self.gui_name.value or "default")
        self._sync_knobs()
        self.dirty.set()

    def _toggle_physics(self):
        on = self.gui_physics.value
        self.phys_viz.set_visible(on)
        self.phys_label.visible = on
        if on:
            self._reset_physics()

    def _reset_physics(self):
        if self.cycle is None:
            return
        frame = self._frame_index()
        self.physics.reset(self.cycle.qpos[frame])
        self.physics_reset_at = None

    def _toggle_clip(self):
        on = self.gui_clip.value
        self.clip = None
        if on:
            path = self.gui_clip_path.value
            if not os.path.isabs(path):
                path = os.path.join(HERE, "..", path)
            try:
                data = np.load(path, allow_pickle=True)
                names = [str(n) for n in data["joint_names"]]
                pos = data["positions"]
                fps = float(data["fps"]) if "fps" in data else 1.0 / float(data["dt"])
                cols = np.zeros((pos.shape[0], self.robot.num_joints))
                for j, n in enumerate(self.robot.joint_names):
                    if n in names:
                        cols[:, j] = pos[:, names.index(n)]
                self.clip = (cols, fps)
                self.clip_t = 0.0
            except Exception as exc:  # noqa: BLE001 - show any load problem in the UI
                self.status_md.content = f"Could not load clip: `{exc}`"
                self.gui_clip.value = False
                on = False
        self.clip_viz.set_visible(on)

    # ------------------------------------------------------------------ bake
    def _bake(self):
        self.gui_bake.disabled = True
        self.bake_bar.visible = True
        self.bake_bar.value = 0.0
        out = self.gui_out.value
        with self.lock:
            style = self.style.copy()
            style.name = self.gui_name.value or style.name
        self.bake_md.content = f"Baking **{style.name}**…"
        try:
            res = bake(style, out, self.robot.config_path,
                       progress=lambda n, tot: setattr(self.bake_bar, "value", 100.0 * n / tot))
            st = res["status"]
            self.bake_md.content = (f"Wrote `{out}`\n\n{ICON[OK]} {int((st == 0).sum())} "
                                    f"{ICON[WARN]} {int((st == 1).sum())} {ICON[FAIL]} {int((st == 2).sum())} "
                                    f"of {st.size} commands, period {1.0 / res['freq'].max():.2f}-"
                                    f"{1.0 / res['freq'].min():.2f} s\n\n"
                                    f"```\n{res['report']}\n```")
        except Exception:  # noqa: BLE001
            self.bake_md.content = f"Bake failed:\n```\n{traceback.format_exc()}\n```"
        self.gui_bake.disabled = False
        self.bake_bar.visible = False

    # ------------------------------------------------------------------ regeneration
    def _regen_loop(self):
        while True:
            self.dirty.wait()
            self.dirty.clear()
            try:
                t = time.time()
                with self.lock:
                    style = self.style.copy()
                cmd = self.cmd.copy()
                thr = self.robot.stand_threshold
                stopped = np.hypot(cmd[0], cmd[1]) < thr and abs(cmd[2]) < thr
                if stopped and not self.gui_step_still.value:
                    cyc = self.gen.stand_cycle(style)
                else:
                    cyc = self.gen.cycle(style, cmd)
                rep = self.analyzer.analyze(cyc)
                _, stand_q = self.gen.stand(style)
                self.gen_ms = 1000 * (time.time() - t)
                self._swap_cycle(cyc, rep, stand_q)
            except Exception:  # noqa: BLE001
                self.status_md.content = f"Generation failed:\n```\n{traceback.format_exc()}\n```"

    def _swap_cycle(self, cyc, rep, stand_q):
        with self.lock:
            old = self.cycle
            phase = self.play_t / old.period % 1.0 if old is not None else 0.0
            if old is not None:
                # keep the base where it is on screen across the swap
                f_old = int(phase * old.n_frames) % old.n_frames
                f_new = int(phase * cyc.n_frames) % cyc.n_frames
                self.offset = _compose(_compose(self.offset, _base_planar(old.qpos[f_old])),
                                       _inverse(_base_planar(cyc.qpos[f_new])))
            self.cycle, self.report, self.stand_q = cyc, rep, stand_q
            if self.shown_q is not None:
                self.blend_from = self.shown_q.copy()
                self.blend_t0 = time.time()
            self.play_t = phase * cyc.period
            self.last_frame = int(phase * cyc.n_frames) % cyc.n_frames
            self.loop_disp = _compose(_base_planar(cyc.qpos_ext[-1]), _inverse(_base_planar(cyc.qpos_ext[1])))
        self._draw_static_overlays()
        self._update_status()
        self._update_plots()

    def _update_status(self):
        cyc, rep = self.cycle, self.report
        if cyc is None:
            return
        st = cyc.style
        if cyc.standing:
            hint = (" · tick *Step in place when stopped* to preview the Base gait"
                    if self._current_anchor() is None else "")
            self.status_md.content = (f"{ICON[rep.status]} **STANDING** · zero command, clock holds (as in training)"
                                      f"{hint} · {self.gen_ms:.0f} ms")
            rows = ["| check | value | |", "|---|---|---|"]
            for c in rep.checks:
                rows.append(f"| {c.name} | {c.value:.2f} {c.unit} {('· ' + c.detail) if c.detail else ''} | {ICON[c.status]} |")
            self.check_md.content = "\n".join(rows)
            return
        step = math.hypot(self.cmd[0], self.cmd[1]) * cyc.period / 2
        froude = (self.cmd[0] ** 2 + self.cmd[1] ** 2) / (9.81 * self.robot.leg_length)
        heel = ""
        if cyc.pitch_profiles:
            _, p = cyc.pitch_profiles["right"]
            heel = f" · land {-math.degrees(p[0]):+.0f}° / leave {math.degrees(p[-1]):+.0f}° (toes up / heel up)"
        self.status_md.content = (
            f"{ICON[rep.status]} **{rep.status.upper()}** · cmd ({self.cmd[0]:+.2f}, {self.cmd[1]:+.2f}, "
            f"{self.cmd[2]:+.2f}) · cycle {cyc.period:.2f} s ({120.0 / cyc.period:.0f} steps/min) · step "
            f"{step * 100:.1f} cm · Fr {froude:.3f} · width {st['stance_width'] * 100:.1f} cm{heel} · "
            f"{self.gen_ms:.0f} ms")
        rows = ["| check | value | |", "|---|---|---|"]
        for c in rep.checks:
            rows.append(f"| {c.name} | {c.value:.2f} {c.unit} {('· ' + c.detail) if c.detail else ''} | {ICON[c.status]} |")
        self.check_md.content = "\n".join(rows)

    def _update_plots(self):
        cyc, rep = self.cycle, self.report
        if cyc is None:
            return
        j = self.robot.joint_names.index(self.gui_joint.value)
        n = cyc.n_frames
        x = np.arange(n) / n
        ang = np.degrees(cyc.field("joint_pos")[:, j])
        self.plot_angle.data = (x, ang, np.full(n, np.degrees(self.robot.lower[j])),
                                np.full(n, np.degrees(self.robot.upper[j])))
        self.plot_angle.title = f"{self.gui_joint.value} angle (°)"
        self.plot_speed.data = (x, np.abs(cyc.field("joint_vel")[:, j]), np.full(n, min(self.robot.actuator.no_load_speed, self.robot.actuator.slew_limit)))
        use = rep.envelope_usage * 100
        self.plot_usage.data = (x, use[:, j], use.max(axis=1), np.full(n, 100.0))
        fh = cyc.field("foot_height") * 1000
        self.plot_feet.data = (x, fh[:, 0], fh[:, 1], rep.zmp_margin * 1000)

    # ------------------------------------------------------------------ render
    def _frame_index(self):
        cyc = self.cycle
        if self.gui_pause.value:
            return int(self.gui_phase.value * cyc.n_frames) % cyc.n_frames
        return int(self.play_t / cyc.dt) % cyc.n_frames

    def _render_loop(self):
        dt_render = 1.0 / 50.0
        tour_t, tour_i = 0.0, 0
        tour = ["Fwd", "Stop", "Left", "Turn R", "Back", "Right", "Turn L", "Stop"]
        while True:
            t0 = time.time()
            if self.cycle is None:
                time.sleep(0.05)
                continue
            speed = self.gui_speed.value
            try:
                with self.lock:
                    cyc = self.cycle
                    if not self.gui_pause.value:
                        self.play_t += dt_render * speed
                    f = self._frame_index()
                    if f < self.last_frame and not self.gui_pause.value:   # wrapped -> next loop
                        self.offset = _compose(self.offset, self.loop_disp)
                        if np.hypot(self.offset[0], self.offset[1]) > 0.8:
                            self.offset = _planar(0.0, 0.0, self.offset[2])
                    self.last_frame = f
                    if self.gui_inplace.value and cyc.path is not None:
                        disp_T = _inverse(cyc.path[f])      # cancel the commanded travel
                    else:
                        disp_T = self.offset.copy()

                    q_show = cyc.qpos[f].copy()
                    if self.blend_from is not None:
                        a = (time.time() - self.blend_t0) / self.blend_time
                        if a >= 1.0:
                            self.blend_from = None
                        else:
                            a = a * a * (3.0 - 2.0 * a)
                            j = self.robot.qpos_adr
                            q_show[j] = (1.0 - a) * self.blend_from[j] + a * q_show[j]
                            q_show[2] = (1.0 - a) * self.blend_from[2] + a * q_show[2]
                    self.shown_q = q_show
                with self.server.atomic():
                    self.disp.position = (disp_T[0], disp_T[1], 0.0)
                    self.disp.wxyz = _quat_z(disp_T[2])
                    self.robot_viz.update(q_show)
                    show_bal = self.gui_balance.value
                    com = cyc.com[f]
                    self.com_h.position = com
                    self.com_ground_h.position = (com[0], com[1], 0.001)
                    z = cyc.zmp[f]
                    self.zmp_h.position = (z[0], z[1], 0.002)
                    for h in (self.com_h, self.com_ground_h, self.zmp_h):
                        h.visible = show_bal
                    poly = cyc.support[f]
                    pts = np.concatenate([poly, poly[:1]])
                    seg = np.stack([pts[:-1], pts[1:]], axis=1)
                    seg3 = np.concatenate([seg, np.full(seg.shape[:2] + (1,), 0.0015)], axis=-1)
                    if self.support_h is not None:
                        self.support_h.remove()
                    self.support_h = self.server.scene.add_line_segments(
                        "/disp/overlay/support", seg3, (40, 200, 90), thickness=2.0, thickness_units="screen",
                        visible=show_bal)

                    if self.gui_physics.value:
                        target = cyc.field("joint_pos")[f]
                        self.physics.step(target, dt_render * speed)
                        self.phys_viz.update(self.physics.qpos, _planar(0.0, -0.35, 0.0))
                        pq = self.physics.qpos
                        self.phys_label.position = (pq[0], pq[1] - 0.35, 0.33)
                        self.phys_label.text = "physics: FELL" if self.physics.fallen else "physics"
                        if self.physics.fallen:
                            if self.physics_reset_at is None:
                                self.physics_reset_at = time.time() + 1.5
                            elif time.time() > self.physics_reset_at:
                                self._reset_physics()

                    if self.clip is not None and self.stand_q is not None:
                        cols, fps = self.clip
                        self.clip_t += dt_render * speed * self.gui_clip_speed.value
                        q = self.stand_q.copy()
                        q[self.robot.qpos_adr] = cols[int(self.clip_t * fps) % len(cols)]
                        q[0:2] = 0.0
                        self.clip_viz.update(q, _planar(0.0, 0.35, 0.0))

                if self.gui_tour.value:
                    tour_t += dt_render
                    if tour_t > 4.0:
                        tour_t, tour_i = 0.0, (tour_i + 1) % len(tour)
                        self._set_preset(tour[tour_i])
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                time.sleep(0.5)
            time.sleep(max(0.0, dt_render - (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser(description="Gait Studio")
    ap.add_argument("--robot", default=DEFAULT_ROBOT_CONFIG)
    ap.add_argument("--style", default=None, help="style yaml to start from (default: styles/default.yaml)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach the Studio from other machines")
    args = ap.parse_args()
    Studio(args.robot, args.style, args.port, args.host)
    print(f"\n  Gait Studio running: open http://localhost:{args.port}\n")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
