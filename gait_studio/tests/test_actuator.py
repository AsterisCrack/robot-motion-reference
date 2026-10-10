"""BAM actuator model. Run from robot_motion_reference/:  python -m pytest gait_studio/tests -q"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from gait_studio.actuator import BamActuator
from gait_studio.analysis import Analyzer
from gait_studio.gait import GaitGenerator
from gait_studio.robot import RobotModel
from gait_studio.style import Style

BIPED_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def robot():
    return RobotModel()


def test_stall_cap_and_speeds(robot):
    a = robot.actuator
    assert a.stall_torque == pytest.approx(2.94, rel=1e-6)
    # the cap raises R only: no-load speed is the uncapped V/kt
    assert a.no_load_speed == pytest.approx(0.97 * 12.0 / a.p.kt, rel=1e-9)
    assert robot.armature == pytest.approx(a.p.armature)


def test_envelope(robot):
    a = robot.actuator
    k_e = a.p.kt ** 2 / a.p.R
    assert a.torque_limit(np.array(1.0), np.array(0.0)) == pytest.approx(a.stall_torque)
    assert a.torque_limit(np.array(1.0), np.array(a.no_load_speed)) == pytest.approx(
        a.stall_torque - k_e * a.no_load_speed)
    # driving at speed loses torque, braking at the same speed gains it
    drive = a.torque_limit(np.array(1.0), np.array(2.0))
    brake = a.torque_limit(np.array(-1.0), np.array(2.0))
    assert drive < a.stall_torque < brake


def test_friction_always_opposes_gain(robot):
    a = robot.actuator
    tau = np.array([0.5, -0.5, 0.5, 0.0])
    dq = np.array([1.0, -1.0, -1.0, 0.0])
    m = a.required_motor_torque(tau, dq)
    assert m[0] > tau[0] and m[1] < tau[1]      # moving: the motor also pays friction
    assert m[2] < tau[2]                         # braking: friction helps
    assert m[3] == pytest.approx(0.0)


def test_stand_is_cheap_and_overspeed_fails(robot):
    gen, an = GaitGenerator(robot), Analyzer(robot)
    stand = an.analyze(gen.cycle(Style(), (0.0, 0.0, 0.0)))
    usage = {c.name: c.value for c in stand.checks}["torque-speed usage"]
    assert usage < 100.0
    fast = an.analyze(gen.cycle(Style(), (0.3, 0.0, 0.0)))
    assert fast.status != stand.status or {c.name: c.value for c in fast.checks}["torque-speed usage"] > 100.0


def test_numpy_matches_torch_kernel(robot):
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(BIPED_ROOT))
    try:
        from actuator.bam_kernel import BamKernel, apply_stall_cap, load_params
    finally:
        sys.path.pop(0)
    cfg = robot.cfg["actuator"]
    params = apply_stall_cap(load_params(robot.actuator.p.source), cfg["stall_torque_cap_nm"])
    nj, dt, steps = 12, 0.005, 200
    kernel = BamKernel(params, 1, nj, dt, kp=cfg["kp_fw"], vin=cfg["vin"], dtype=torch.float64)
    act = BamActuator.from_config(cfg, Path(robot.config_path).parent)

    rng = np.random.default_rng(0)
    q = rng.uniform(-0.3, 0.3, nj)
    for _ in range(steps):
        goal = q + rng.uniform(-0.4, 0.4, nj)
        dq = rng.uniform(-3.0, 3.0, nj)
        ext = rng.uniform(-1.5, 1.5, nj)
        inertia = rng.uniform(0.02, 0.1, nj)
        tm_t = kernel.motor(torch.tensor(goal)[None], torch.tensor(q)[None], torch.tensor(dq)[None])
        tf_t = kernel.stop_clip(tm_t, torch.tensor(ext)[None], torch.tensor(dq)[None], torch.tensor(inertia)[None])
        tm = act.motor(goal, q, dq, dt)
        tf = act.stop_clip(tm, ext, dq, inertia, dt)
        np.testing.assert_allclose(tm, tm_t[0].numpy(), atol=1e-9)
        np.testing.assert_allclose(tf, tf_t[0].numpy(), atol=1e-9)
        q = q + 0.01 * rng.standard_normal(nj)
