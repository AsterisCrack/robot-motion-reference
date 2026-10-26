"""Bake a style into a gait table for training.

    python -m gait_studio.bake --style gait_studio/styles/default.yaml --out processed/gait_table.npz

Writes one steady-state cycle per (vx, vy, wz) grid point plus a standing frame. The env
(utils/motion_reference.py GaitTableReference) interpolates the table trilinearly in the
command and linearly in phase.

Table v3 adds the command limit the bake used (``command_limit_shape`` from configs/robot_v2.yaml,
``command_ranges_x/y/z``) and ``cmd_baked`` [nvx, nvy, nwz, 3]: the command each cell was actually baked
at. With shape ``ellipse_xy`` a cell outside the (vx, vy) ellipse is baked at its projected command, so
the trilinear blend near the boundary is feasible. This only mirrors the training config's
``commands.base_velocity.limit_shape``: training checks the metadata and never reads the limit from the table.

Table v2: every cycle has its own period (the cadence law, cadence.py), so cycles are stored
phase-normalised (n_phase samples each) next to a `freq` grid [nvx, nvy, nwz] of gait clock
frequencies in cycles/s. Every runtime advances its clock by 2*pi*dt*freq(cmd), with freq
interpolated trilinearly in the command and clamped to the grid, the same rule as the table.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

# Each worker process solves many tiny (<20x20) linear systems in ik.py's IK loop. OpenBLAS's
# multithreading overhead per call dwarfs the work for matrices this small, and it's far worse
# on Windows than macOS/Linux -- combined with one BLAS thread pool per pool worker, a many-core
# Windows box ends up thread-thrashing instead of going faster. Pin to 1 thread per process
# before numpy is imported anywhere (here or in the worker's re-import under spawn) and let the
# process pool provide the parallelism instead.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import yaml

from .analysis import FAIL, OK, WARN, Analyzer
from .command_limit import limit_command, validate_shape
from .gait import FIELDS, FRAME_DIM, GaitGenerator, mirror_rows
from .robot import DEFAULT_ROBOT_CONFIG, RobotModel
from .style import Style

# 3 = 2 + command-limit metadata (command_limit_shape, command_ranges_x/y/z, cmd_baked). Readers accept 1-3.
TABLE_VERSION = 3
PHASE_CONVENTION = "phase 0 = right-foot touchdown; [0, 0.5) right stance, [0.5, 1) left stance"
_STATUS_CODE = {OK: 0, WARN: 1, FAIL: 2}

_worker = {}


def _init_worker(robot_cfg: str, style_dict: dict):
    robot = RobotModel(robot_cfg)
    _worker["gen"] = GaitGenerator(robot)
    _worker["an"] = Analyzer(robot)
    _worker["style"] = Style.from_dict(style_dict)


def _bake_point(args):
    idx, cmd = args
    cyc = _worker["gen"].cycle(_worker["style"], cmd)
    rep = _worker["an"].analyze(cyc)
    checks = {c.name: c.value for c in rep.checks}
    return idx, cyc.frames.astype(np.float32), cyc.freq, _STATUS_CODE[rep.status], checks, rep.summary()


def symmetrize_table(data: np.ndarray, axes: dict, robot: RobotModel) -> np.ndarray:
    """Make reference(vx, -vy, -wz, phase + 0.5) exactly the mirror of reference(vx, vy, wz, phase).

    The generator is symmetric only up to the robot model's own asymmetry (the base CoM sits
    1.7 mm off-centre, ~1 deg of joint difference). The env's symmetry augmentation mirrors
    observations and reuses the reward, so the table must be exactly symmetric.
    """
    for k in ("vy", "wz"):
        if not np.allclose(axes[k], -axes[k][::-1], atol=1e-6):
            print(f"  note: {k} grid is not symmetric about 0, table left unsymmetrized")
            return data
    n = data.shape[3]
    partner = np.roll(data[:, ::-1, ::-1], -n // 2, axis=3)   # (vx, -vy, -wz), phase + 0.5
    return (0.5 * (data + mirror_rows(partner, robot))).astype(np.float32)


def symmetrize_freq(freq: np.ndarray, axes: dict) -> np.ndarray:
    """freq(vx, -vy, -wz) == freq(vx, vy, wz) exactly, so the mirrored clock runs at the same rate."""
    for k in ("vy", "wz"):
        if not np.allclose(axes[k], -axes[k][::-1], atol=1e-6):
            return freq
    return (0.5 * (freq + freq[:, ::-1, ::-1])).astype(np.float32)


def grid_axes(robot: RobotModel, grid: dict | None = None) -> dict[str, np.ndarray]:
    grid = grid or robot.cfg.get("grid", {"vx": 9, "vy": 7, "wz": 9})
    return {k: np.linspace(*robot.command_ranges[k], int(grid[k])) for k in ("vx", "vy", "wz")}


def baked_commands(axes: dict, robot: RobotModel, limit_shape: str) -> np.ndarray:
    """The command each grid cell is baked at, [nvx, nvy, nwz, 3]: the cell's own command, projected into the
    command limit (robot.command_ranges + limit_shape). Identity for shape 'box'."""
    cell = np.stack(np.meshgrid(axes["vx"], axes["vy"], axes["wz"], indexing="ij"), axis=-1)
    return limit_command(cell, robot.limit_ranges, limit_shape)


def bake(style: Style, out_path: str, robot_cfg: str = DEFAULT_ROBOT_CONFIG, grid: dict | None = None,
         workers: int | None = None, progress=None, limit_shape: str | None = None) -> dict:
    robot = RobotModel(robot_cfg)
    gen = GaitGenerator(robot)
    axes = grid_axes(robot, grid)
    shape = tuple(len(axes[k]) for k in ("vx", "vy", "wz"))
    limit_shape = robot.command_limit_shape if limit_shape is None else validate_shape(limit_shape)
    cell_cmd = np.stack(np.meshgrid(axes["vx"], axes["vy"], axes["wz"], indexing="ij"), axis=-1)
    cmd_baked = baked_commands(axes, robot, limit_shape)
    projected = np.abs(cmd_baked - cell_cmd).max(axis=-1) > 1e-9          # [nvx, nvy, nwz]
    jobs = [((i, j, k), tuple(cmd_baked[i, j, k]))
            for i in range(shape[0]) for j in range(shape[1]) for k in range(shape[2])]

    probe = gen.cycle(style, (0.0, 0.0, 0.0))
    n_frames = probe.n_frames
    data = np.zeros(shape + (n_frames, FRAME_DIM), dtype=np.float32)
    freq = np.zeros(shape, dtype=np.float32)
    status = np.zeros(shape, dtype=np.int8)
    metrics = {name: np.zeros(shape, dtype=np.float32) for name in
               ("peak joint speed", "peak joint torque", "torque-speed usage", "ZMP margin",
                "min foot gap", "joint limit margin", "IK foot/CoM error")}

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(robot.config_path, style.to_dict())) as pool:
        for n, (idx, frames, f, st, checks, _) in enumerate(pool.imap_unordered(_bake_point, jobs, chunksize=4)):
            data[idx] = frames
            freq[idx] = f
            status[idx] = st
            for name in metrics:
                metrics[name][idx] = checks.get(name, np.nan)
            if progress:
                progress(n + 1, len(jobs))

    data = symmetrize_table(data, axes, robot)
    freq = symmetrize_freq(freq, axes)
    stand_frame, _ = gen.stand(style)
    stand_frame = 0.5 * (stand_frame + mirror_rows(stand_frame, robot))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(
        out_path,
        version=np.int32(TABLE_VERSION),
        data=data,
        stand=stand_frame.astype(np.float32),
        grid_vx=axes["vx"].astype(np.float32),
        grid_vy=axes["vy"].astype(np.float32),
        grid_wz=axes["wz"].astype(np.float32),
        field_names=np.array([f for f, _ in FIELDS]),
        field_widths=np.array([w for _, w in FIELDS], dtype=np.int32),
        joint_names=np.array(robot.joint_names),
        freq=freq,                                   # gait clock [cycles/s] per command
        period=np.float32(probe.period),             # zero-command cycle, for display only
        control_dt=np.float32(robot.control_dt),
        n_frames=np.int32(n_frames),                 # samples per cycle (phase-normalised)
        phase_convention=np.array(PHASE_CONVENTION),
        command_limit_shape=np.array(limit_shape),   # v3: what this bake mirrored (training's config is the truth)
        command_ranges_x=np.array(robot.command_ranges["vx"], dtype=np.float64),
        command_ranges_y=np.array(robot.command_ranges["vy"], dtype=np.float64),
        command_ranges_z=np.array(robot.command_ranges["wz"], dtype=np.float64),
        cmd_baked=cmd_baked,                         # [nvx, nvy, nwz, 3]: command each cell was baked at
        status=status,
        **{"metric_" + k.replace(" ", "_").replace("/", "_").replace("-", "_"): v for k, v in metrics.items()},
        style_yaml=np.array(yaml.safe_dump(style.to_dict(), sort_keys=False)),
        style_digest=np.array(style.digest()),
    )
    style.save(os.path.splitext(out_path)[0] + "_style.yaml")
    report = _report(axes, status, metrics, freq, n_frames, time.time() - t0, limit_shape, projected)
    with open(os.path.splitext(out_path)[0] + "_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    return {"path": out_path, "shape": data.shape, "status": status, "metrics": metrics, "report": report,
            "period": probe.period, "freq": freq, "limit_shape": limit_shape, "projected": projected,
            "cmd_baked": cmd_baked}


def _status_counts(status: np.ndarray) -> str:
    return (f"ok {int((status == 0).sum())}  warn {int((status == 1).sum())}  fail {int((status == 2).sum())}"
            f"  (of {status.size})")


def _report(axes, status, metrics, freq, n_frames, seconds, limit_shape="box", projected=None) -> str:
    n = status.size
    period = 1.0 / freq
    lines = [
        f"Gait table: {status.shape} commands x {n_frames} samples per cycle, "
        f"period {period.min():.2f}-{period.max():.2f} s, baked in {seconds:.0f} s",
        f"  ok {int((status == 0).sum())}/{n}   warn {int((status == 1).sum())}/{n}   fail {int((status == 2).sum())}/{n}",
        f"Command limit (mirrors the training config): {limit_shape}",
    ]
    if projected is not None and limit_shape != "box":
        lines += [f"  cells baked at a projected command (outside the limit): {int(projected.sum())}/{n}",
                  f"  inside the limit : {_status_counts(status[~projected])}",
                  f"  projected cells  : {_status_counts(status[projected])}"]
    lines += [
        "",
        "Status map at wz = 0 (rows vx, cols vy): . ok, ~ warn, X fail",
    ]
    k0 = int(np.argmin(np.abs(axes["wz"])))
    lines.append("   vx\\vy " + " ".join(f"{v:+.2f}" for v in axes["vy"]))
    for i, vx in enumerate(axes["vx"]):
        row = " ".join({0: "  .  ", 1: "  ~  ", 2: "  X  "}[int(status[i, j, k0])] for j in range(len(axes["vy"])))
        lines.append(f"  {vx:+.2f}  {row}")
    lines.append("")
    lines.append("Cycle period s at wz = 0 (rows vx, cols vy):")
    lines.append("   vx\\vy " + " ".join(f"{v:+.2f}" for v in axes["vy"]))
    for i, vx in enumerate(axes["vx"]):
        lines.append(f"  {vx:+.2f}  " + " ".join(f"{period[i, j, k0]:5.2f}" for j in range(len(axes["vy"]))))
    lines.append("")
    j0 = int(np.argmin(np.abs(axes["vy"])))
    lines.append("Cycle period s at vy = 0 (rows vx, cols wz):")
    lines.append("   vx\\wz " + " ".join(f"{w:+.2f}" for w in axes["wz"]))
    for i, vx in enumerate(axes["vx"]):
        lines.append(f"  {vx:+.2f}  " + " ".join(f"{period[i, j0, k]:5.2f}" for k in range(len(axes["wz"]))))
    lines.append("")
    lines.append("Torque-speed usage % at vy = 0 (rows vx, cols wz):")
    lines.append("   vx\\wz " + " ".join(f"{w:+.2f}" for w in axes["wz"]))
    for i, vx in enumerate(axes["vx"]):
        lines.append(f"  {vx:+.2f}  " + " ".join(f"{metrics['torque-speed usage'][i, j0, k]:5.0f}" for k in range(len(axes["wz"]))))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--style", default=os.path.join(here, "styles", "default.yaml"))
    ap.add_argument("--robot", default=DEFAULT_ROBOT_CONFIG)
    ap.add_argument("--out", default=os.path.join(here, "..", "processed", "gait_table.npz"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--grid", type=int, nargs=3, metavar=("NVX", "NVY", "NWZ"), default=None)
    ap.add_argument("--limit-shape", choices=["box", "ellipse_xy"], default=None,
                    help="override configs/robot_v2.yaml command_limit_shape for this bake")
    args = ap.parse_args()

    style = Style.load(args.style) if os.path.exists(args.style) else Style()
    grid = dict(zip(("vx", "vy", "wz"), args.grid)) if args.grid else None

    def progress(n, total):
        if n % 25 == 0 or n == total:
            print(f"\r  {n}/{total}", end="", flush=True)

    print(f"Baking style '{style.name}' ({style.digest()}) -> {os.path.abspath(args.out)}")
    res = bake(style, args.out, args.robot, grid, args.workers, progress, args.limit_shape)
    print()
    print(res["report"])


if __name__ == "__main__":
    main()
