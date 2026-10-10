"""Bake a style into a gait table for training.

    python -m gait_studio.bake --style gait_studio/styles/default.yaml --out processed/gait_table.npz

Writes one steady-state cycle per (vx, vy, wz) grid point plus a standing frame. The env
(utils/motion_reference.py GaitTableReference) interpolates the table trilinearly in the
command and linearly in phase.
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
from .gait import FIELDS, FRAME_DIM, GaitGenerator, mirror_rows
from .robot import DEFAULT_ROBOT_CONFIG, RobotModel
from .style import Style

TABLE_VERSION = 1
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
    return idx, cyc.frames.astype(np.float32), _STATUS_CODE[rep.status], checks, rep.summary()


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


def grid_axes(robot: RobotModel, grid: dict | None = None) -> dict[str, np.ndarray]:
    grid = grid or robot.cfg.get("grid", {"vx": 9, "vy": 7, "wz": 9})
    return {k: np.linspace(*robot.command_ranges[k], int(grid[k])) for k in ("vx", "vy", "wz")}


def bake(style: Style, out_path: str, robot_cfg: str = DEFAULT_ROBOT_CONFIG, grid: dict | None = None,
         workers: int | None = None, progress=None) -> dict:
    robot = RobotModel(robot_cfg)
    gen = GaitGenerator(robot)
    axes = grid_axes(robot, grid)
    shape = tuple(len(axes[k]) for k in ("vx", "vy", "wz"))
    jobs = [((i, j, k), (axes["vx"][i], axes["vy"][j], axes["wz"][k]))
            for i in range(shape[0]) for j in range(shape[1]) for k in range(shape[2])]

    probe = gen.cycle(style, (0.0, 0.0, 0.0))
    n_frames = probe.n_frames
    data = np.zeros(shape + (n_frames, FRAME_DIM), dtype=np.float32)
    status = np.zeros(shape, dtype=np.int8)
    metrics = {name: np.zeros(shape, dtype=np.float32) for name in
               ("peak joint speed", "peak joint torque", "torque-speed usage", "ZMP margin",
                "min foot gap", "joint limit margin", "IK foot/CoM error")}

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(robot.config_path, style.to_dict())) as pool:
        for n, (idx, frames, st, checks, _) in enumerate(pool.imap_unordered(_bake_point, jobs, chunksize=4)):
            data[idx] = frames
            status[idx] = st
            for name in metrics:
                metrics[name][idx] = checks.get(name, np.nan)
            if progress:
                progress(n + 1, len(jobs))

    data = symmetrize_table(data, axes, robot)
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
        period=np.float32(probe.period),
        dt=np.float32(robot.control_dt),
        n_frames=np.int32(n_frames),
        phase_convention=np.array(PHASE_CONVENTION),
        status=status,
        **{"metric_" + k.replace(" ", "_").replace("/", "_").replace("-", "_"): v for k, v in metrics.items()},
        style_yaml=np.array(yaml.safe_dump(style.to_dict(), sort_keys=False)),
        style_digest=np.array(style.digest()),
    )
    style.save(os.path.splitext(out_path)[0] + "_style.yaml")
    report = _report(axes, status, metrics, probe.period, time.time() - t0)
    with open(os.path.splitext(out_path)[0] + "_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    return {"path": out_path, "shape": data.shape, "status": status, "metrics": metrics, "report": report,
            "period": probe.period}


def _report(axes, status, metrics, period, seconds) -> str:
    n = status.size
    lines = [
        f"Gait table: {status.shape} commands x {int(round(period / 0.02))} frames, period {period:.2f} s, baked in {seconds:.0f} s",
        f"  ok {int((status == 0).sum())}/{n}   warn {int((status == 1).sum())}/{n}   fail {int((status == 2).sum())}/{n}",
        "",
        "Status map at wz = 0 (rows vx, cols vy): . ok, ~ warn, X fail",
    ]
    k0 = int(np.argmin(np.abs(axes["wz"])))
    lines.append("   vx\\vy " + " ".join(f"{v:+.2f}" for v in axes["vy"]))
    for i, vx in enumerate(axes["vx"]):
        row = " ".join({0: "  .  ", 1: "  ~  ", 2: "  X  "}[int(status[i, j, k0])] for j in range(len(axes["vy"])))
        lines.append(f"  {vx:+.2f}  {row}")
    lines.append("")
    lines.append("Torque-speed usage % at vy = 0 (rows vx, cols wz):")
    j0 = int(np.argmin(np.abs(axes["vy"])))
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
    args = ap.parse_args()

    style = Style.load(args.style) if os.path.exists(args.style) else Style()
    grid = dict(zip(("vx", "vy", "wz"), args.grid)) if args.grid else None

    def progress(n, total):
        if n % 25 == 0 or n == total:
            print(f"\r  {n}/{total}", end="", flush=True)

    print(f"Baking style '{style.name}' ({style.digest()}) -> {os.path.abspath(args.out)}")
    res = bake(style, args.out, args.robot, grid, args.workers, progress)
    print()
    print(res["report"])


if __name__ == "__main__":
    main()
