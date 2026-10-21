"""Gait generator invariants. Run from robot_motion_reference/:  python -m pytest gait_studio/tests -q"""
from __future__ import annotations

import numpy as np
import pytest

from gait_studio.bake import bake
from gait_studio.gait import FIELD_SLICES, GaitGenerator, mirror_rows
from gait_studio.robot import RobotModel
from gait_studio.style import Style


@pytest.fixture(scope="module")
def gen():
    return GaitGenerator(RobotModel())


MIRROR_CMDS = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.05, 0.08, 0.0), (0.0, 0.0, 0.3), (0.1, -0.05, 0.2)]


def _mirror_error(gen, style, cmd):
    a = gen.cycle(style, cmd)
    b = gen.cycle(style, (cmd[0], -cmd[1], -cmd[2]))
    n = a.n_frames
    shifted = np.roll(b.frames, -n // 2, axis=0)
    return np.abs(mirror_rows(a.frames, gen.r) - shifted)


@pytest.mark.parametrize("cmd", MIRROR_CMDS)
def test_mirror_symmetry(gen, cmd):
    """reference(mirrored cmd, phase + 0.5) ~= mirror(reference(cmd, phase)).

    Holds up to the model's own asymmetry (base CoM 1.7 mm off-centre); bake() then makes the
    table exactly symmetric, see test_baked_table_is_symmetric. Checked at a cycle of at most
    0.9 s: slower cycles mean longer single support, which amplifies that asymmetry (below).
    """
    style = Style()
    style.base["period_max"] = 0.9
    err = _mirror_error(gen, style, cmd)
    F = FIELD_SLICES
    assert err[:, F["joint_pos"]].max() < np.radians(1.5)
    assert err[:, F["contact"]].max() < 1e-6
    assert err[:, F["foot_pos_b"]].max() < 3e-3


@pytest.mark.parametrize("cmd", MIRROR_CMDS)
def test_mirror_symmetry_human_cadence(gen, cmd):
    """Same with the default cadence (cycles up to 2 s): looser, the 1.7 mm CoM offset acts longer."""
    err = _mirror_error(gen, Style(), cmd)
    F = FIELD_SLICES
    assert err[:, F["joint_pos"]].max() < np.radians(4.0)
    assert err[:, F["contact"]].max() < 1e-6
    assert err[:, F["foot_pos_b"]].max() < 5e-3


@pytest.mark.parametrize("cmd", [(0.0, 0.0, 0.0), (0.1, 0.05, 0.2)])
def test_periodic(gen, cmd):
    """The frame after the last one closes the loop onto frame 0."""
    c = gen.cycle(Style(), cmd)
    qa = c.qpos_ext[-1][gen.r.qpos_adr]
    qb = c.qpos[0][gen.r.qpos_adr]
    assert np.abs(qa - qb).max() < np.radians(0.5)


def test_tracks_command(gen):
    """Mean base velocity over a cycle equals the command."""
    cmd = (0.1, 0.05, 0.2)
    c = gen.cycle(Style(), cmd)
    v = c.field("base_lin_vel_b").mean(axis=0)
    w = c.field("base_ang_vel_b").mean(axis=0)
    assert abs(v[0] - cmd[0]) < 0.01 and abs(v[1] - cmd[1]) < 0.01 and abs(w[2] - cmd[2]) < 0.02


def test_phase_convention(gen):
    """Phase [0, 0.5) is right stance: the left foot is airborne mid first half."""
    c = gen.cycle(Style(), (0.0, 0.0, 0.0))
    n = c.n_frames
    contact = c.field("contact")
    assert contact[n // 4, 0] == 0 and contact[n // 4, 1] == 1
    assert contact[3 * n // 4, 0] == 1 and contact[3 * n // 4, 1] == 0


def test_stand_has_both_feet_down(gen):
    frame, _ = gen.stand(Style())
    assert np.all(frame[FIELD_SLICES["contact"]] == 1)
    assert np.abs(frame[FIELD_SLICES["foot_height"]]).max() < 1e-3


def test_baked_table_is_symmetric(tmp_path):
    out = tmp_path / "table.npz"
    bake(Style(), str(out), grid={"vx": 2, "vy": 3, "wz": 3}, workers=2)
    t = np.load(out)
    robot = RobotModel()
    data = t["data"]
    n = data.shape[3]
    assert int(t["version"]) == 2 and n == robot.n_phase == int(t["n_frames"])
    partner = np.roll(data[:, ::-1, ::-1], -n // 2, axis=3)
    assert np.abs(mirror_rows(partner, robot) - data).max() < 1e-5
    freq = t["freq"]
    assert freq.shape == data.shape[:3] and (freq > 0).all()
    assert np.array_equal(freq, freq[:, ::-1, ::-1])       # the mirrored clock runs at the same rate
    st = Style().base                                        # cadence varies, within [period_min, period_max]
    assert np.ptp(freq) > 0.1
    assert freq.min() >= 1.0 / st["period_max"] - 1e-5 and freq.max() <= 1.0 / st["period_min"] + 1e-5
    assert np.abs(mirror_rows(t["stand"], robot) - t["stand"]).max() < 1e-6
    assert list(t["field_names"])[:2] == ["joint_pos", "joint_vel"]


def _toe_and_heel_world(robot, q, side):
    """World positions of the sole's front and back edge midpoints for a qpos."""
    robot.forward(q)
    p, R = robot.sole_pose(side)
    s = robot.soles[side]
    return p + R @ np.array([s.front, 0.0, 0.0]), p - R @ np.array([s.back, 0.0, 0.0])


@pytest.mark.parametrize("cmd", [(0.15, 0.0, 0.0), (-0.15, 0.0, 0.0)])
def test_heel_toe_pivots_stay_planted(gen, cmd):
    """While a stance foot is pitched, its pivot edge stays put on the ground (heel edge while
    toes-up, toe edge while heel-up), forward and backward."""
    style = Style()
    style.base.update(push_off=8.0)
    c = gen.cycle(style, cmd)
    st = style.resolve(cmd, gen.r.command_ranges)
    plan = gen._plan(cmd, st)
    plan.set_pitch_profiles(c.pitch_profiles)
    t0 = 3 * plan.period
    groups, pitched = {}, 0          # (footstep, edge) -> world points
    for i in range(c.n_frames):
        t = t0 + i * c.dt
        where = plan._where("left", t)
        if where[0] != "stance":
            continue
        assert c.contact[i, 0] == 1
        pitch = plan.stance_pitch("left", where[2])
        toe, heel = _toe_and_heel_world(gen.r, c.qpos[i], "left")
        if pitch > 0.02:
            groups.setdefault((where[1], "toe"), []).append(toe)
            pitched += 1
        elif pitch < -0.02:
            groups.setdefault((where[1], "heel"), []).append(heel)
            pitched += 1
    assert pitched >= 4
    assert {edge for _, edge in groups} == {"toe", "heel"}
    for pts in groups.values():
        pts = np.array(pts)
        assert np.ptp(pts, axis=0).max() < 1.5e-3      # planted (IK tolerance)
        assert np.abs(pts[:, 2]).max() < 1.5e-3        # on the ground


def test_stand_cycle_is_static(gen):
    c = gen.stand_cycle(Style())
    assert c.standing
    assert np.ptp(c.frames, axis=0).max() < 1e-9
    assert np.all(c.contact == 1)
    assert np.abs(c.field("joint_vel")).max() < 1e-9
