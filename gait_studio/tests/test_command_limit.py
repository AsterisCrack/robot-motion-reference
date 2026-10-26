"""Command limit (numpy copy of BipedRobot/utils/command_limit.py) and its use by the bake.

Run from robot_motion_reference/:  python -m pytest gait_studio/tests/test_command_limit.py -q
tests/data/command_limit_golden.json is the same file as BipedRobot/tests/data/command_limit_golden.json.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from gait_studio.bake import TABLE_VERSION, baked_commands, bake, grid_axes
from gait_studio.command_limit import limit_command, sample_in_limit, validate_shape
from gait_studio.robot import RobotModel
from gait_studio.style import Style

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "command_limit_golden.json")


def test_golden():
    with open(GOLDEN, encoding="utf-8") as f:
        g = json.load(f)
    assert len(g["cases"]) == 52
    for c in g["cases"]:
        got = limit_command(np.asarray(c["cmd"]), g[c["ranges"]], c["shape"])
        assert np.abs(got - np.asarray(c["out"])).max() < 1e-12, c


def test_shape_validation_and_sampling():
    assert validate_shape(None) == "box"
    with pytest.raises(ValueError):
        validate_shape("oval")
    r = {"lin_vel_x": (-0.2, 0.2), "lin_vel_y": (-0.15, 0.15), "ang_vel_z": (-1.2, 1.2)}
    x = sample_in_limit(20000, r, "ellipse_xy", rng=np.random.default_rng(0))
    assert np.allclose(limit_command(x, r, "ellipse_xy"), x, atol=1e-9)


@pytest.fixture(scope="module")
def robot():
    return RobotModel()


def test_robot_mirrors_the_limit_setting(robot):
    assert robot.command_limit_shape == "ellipse_xy"
    assert np.allclose(robot.limit_command((0.2, 0.15, 0.0)), (0.2 / math.sqrt(2), 0.15 / math.sqrt(2), 0.0))
    assert np.allclose(robot.limit_command((0.2, 0.0, 1.2)), (0.2, 0.0, 1.2))
    assert np.allclose(robot.limit_command((0.2, 0.15, 0.0), shape="box"), (0.2, 0.15, 0.0))


def test_baked_commands_project_only_outside_cells(robot):
    axes = grid_axes(robot)
    cell = np.stack(np.meshgrid(axes["vx"], axes["vy"], axes["wz"], indexing="ij"), axis=-1)
    box = baked_commands(axes, robot, "box")
    ell = baked_commands(axes, robot, "ellipse_xy")
    assert np.array_equal(box, cell)
    moved = np.abs(ell - cell).max(axis=-1) > 1e-9
    assert 0 < moved.sum() < moved.size
    vx_b, vy_b = np.where(cell[..., 0] >= 0, 0.2, 0.2), 0.15
    outside = (cell[..., 0] / vx_b) ** 2 + (cell[..., 1] / vy_b) ** 2 > 1 + 1e-9
    assert np.array_equal(moved, outside)
    assert np.array_equal(ell[~moved], cell[~moved])                     # inside cells keep their command exactly
    assert np.array_equal(ell[..., 2], cell[..., 2])                     # wz untouched


def test_bake_records_the_limit_and_projects_corners(tmp_path):
    out = tmp_path / "t.npz"
    res = bake(Style(), str(out), grid={"vx": 2, "vy": 3, "wz": 3}, workers=2)
    z = np.load(out)
    assert TABLE_VERSION == 3 and int(z["version"]) == 3
    assert str(z["command_limit_shape"]) == "ellipse_xy"
    assert np.allclose(z["command_ranges_x"], (-0.2, 0.2)) and np.allclose(z["command_ranges_z"], (-1.2, 1.2))
    cb = z["cmd_baked"]
    assert cb.shape == (2, 3, 3, 3)
    # the 4 (vx = +-0.2, vy = +-0.15) corners of each wz plane are projected, the vy = 0 cells are not
    assert int(res["projected"].sum()) == 4 * 3
    assert np.allclose(cb[1, 2, :, :2], (0.2 / math.sqrt(2), 0.15 / math.sqrt(2)))
    assert np.allclose(cb[1, 1, :, :2], (0.2, 0.0))
    assert "projected" in res["report"] and "ellipse_xy" in res["report"]
    # box bake: nothing projected
    res_box = bake(Style(), str(tmp_path / "b.npz"), grid={"vx": 2, "vy": 3, "wz": 3}, workers=2, limit_shape="box")
    assert int(res_box["projected"].sum()) == 0 and str(np.load(tmp_path / "b.npz")["command_limit_shape"]) == "box"
