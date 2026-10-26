"""Command limit shared by training, the MuJoCo tools, the export and the Jetson runtime.

One definition of "which (vx, vy, wz) commands exist":

* per axis, every command is clamped to its range (lin_vel_x, lin_vel_y, ang_vel_z: [lo, hi]);
* shape ``box``        - that clamp is the whole limit (the historical behaviour);
* shape ``ellipse_xy`` - additionally (vx / vx_b)^2 + (vy / vy_b)^2 <= 1, with vx_b the forward bound
  (hi) for vx >= 0 and the backward bound (-lo) otherwise, likewise vy_b. An over-limit command is
  scaled down along its own direction in (vx, vy), so the direction is kept. wz is untouched.

The TRAINING config (``commands.base_velocity.limit_shape`` + ``ranges``) is the source of truth; the
run config and the exported bundle carry it, Gait Studio only mirrors it for its bakes.

This is the numpy-only copy of BipedRobot/utils/command_limit.py (Gait Studio cannot import that repo);
the maths is identical and ``tests/data/command_limit_golden.json`` (same file in both repos) pins them
to the same numbers. Gait Studio only MIRRORS the limit (configs/robot_v2.yaml ``command_limit_shape``)
so its bakes cover feasible commands; the training config's ``limit_shape`` is the source of truth.
"""
from __future__ import annotations

import numpy as np

SHAPES = ("box", "ellipse_xy")
# Used for a missing range key; the same fallbacks the Isaac command sampler always had.
DEFAULT_RANGES = {"lin_vel_x": (-0.5, 0.5), "lin_vel_y": (-0.25, 0.25), "ang_vel_z": (-1.0, 1.0)}


def validate_shape(shape) -> str:
    """``shape`` as a string, or ValueError. None -> 'box' (an old config without the key)."""
    if shape is None:
        return "box"
    if not isinstance(shape, str) or shape not in SHAPES:
        raise ValueError(f"unknown command limit shape {shape!r}; expected one of {SHAPES}")
    return shape


def _bounds(ranges) -> tuple[float, float, float, float, float, float]:
    out = []
    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        lo, hi = ranges.get(key, DEFAULT_RANGES[key]) if ranges is not None else DEFAULT_RANGES[key]
        out += [float(lo), float(hi)]
    return tuple(out)


def _xy_norm_np(vx, vy, lo_x, hi_x, lo_y, hi_y):
    """s = sqrt((vx / vx_b)^2 + (vy / vy_b)^2), vx_b = hi_x (vx >= 0) or -lo_x; vx, vy inside the box."""
    bx = np.where(vx >= 0, hi_x, -lo_x)
    by = np.where(vy >= 0, hi_y, -lo_y)
    # A zero bound means the box clamp already forced that component to 0: it adds nothing to s.
    tx = np.where(bx > 0, vx / np.where(bx > 0, bx, 1.0), 0.0)
    ty = np.where(by > 0, vy / np.where(by > 0, by, 1.0), 0.0)
    return np.sqrt(tx * tx + ty * ty)


def limit_command(cmd, ranges, shape: str = "box"):
    """Clamp commands (..., 3) = (vx, vy, wz) to the limit (numpy in, numpy out)."""
    shape = validate_shape(shape)
    lo_x, hi_x, lo_y, hi_y, lo_z, hi_z = _bounds(ranges)
    arr = np.asarray(cmd)
    arr = arr.astype(np.float64) if arr.dtype.kind != "f" else arr
    vx = np.clip(arr[..., 0], lo_x, hi_x)
    vy = np.clip(arr[..., 1], lo_y, hi_y)
    wz = np.clip(arr[..., 2], lo_z, hi_z)
    if shape == "ellipse_xy":
        s = np.maximum(_xy_norm_np(vx, vy, lo_x, hi_x, lo_y, hi_y), 1.0)
        vx, vy = vx / s, vy / s
    return np.stack([vx, vy, wz], axis=-1)


def sample_in_limit(n: int, ranges, shape: str = "box", rng=None):
    """n commands (n, 3) drawn uniformly inside the limit.

    (vx, vy) is uniform over the box (``box``) or the ellipse (``ellipse_xy``, by rejection from the
    box: scaling would pile density on the boundary); wz is uniform in its range, independent.

    ``rng``: np.random.Generator or None.
    """
    shape = validate_shape(shape)
    lo_x, hi_x, lo_y, hi_y, lo_z, hi_z = _bounds(ranges)
    rng = np.random.default_rng() if rng is None else rng
    lo = np.array([lo_x, lo_y, lo_z])
    span = np.array([hi_x - lo_x, hi_y - lo_y, hi_z - lo_z])
    out = lo + span * rng.random((n, 3))
    if shape == "ellipse_xy":
        for _ in range(64):
            bad = _xy_norm_np(out[:, 0], out[:, 1], lo_x, hi_x, lo_y, hi_y) > 1.0
            if not bad.any():
                break
            out[bad] = lo + span * rng.random((int(bad.sum()), 3))
        out = limit_command(out, ranges, shape)
    return out
