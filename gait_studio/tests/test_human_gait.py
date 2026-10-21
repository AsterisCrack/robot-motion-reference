"""Human-gait behaviour: speed-dependent cadence, the rocker model of heel-to-toe, and the
direction-aware knobs (pelvis twist, side-step lift). Run from robot_motion_reference/:
python -m pytest gait_studio/tests -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from gait_studio.gait import FIELD_SLICES, GaitGenerator, mat_from_yaw_pitch
from gait_studio.robot import RobotModel
from gait_studio.style import Style


@pytest.fixture(scope="module")
def gen():
    return GaitGenerator(RobotModel())


def _period(gen, style, cmd):
    return gen.period(cmd, style.resolve(cmd, gen.r.command_ranges))


# ---------------------------------------------------------------------- cadence
def test_cadence_follows_speed(gen):
    """Slower walking = longer cycle (and shorter steps), as in humans; clamped to the knobs."""
    style = Style()
    T = [_period(gen, style, (v, 0.0, 0.0)) for v in (0.05, 0.1, 0.15, 0.2)]
    assert all(a > b for a, b in zip(T, T[1:]))
    steps = [v * t / 2 for v, t in zip((0.05, 0.1, 0.15, 0.2), T)]
    assert all(a < b for a, b in zip(steps, steps[1:]))
    # Human law scaled to this leg: ~0.9 s cycle and ~9 cm steps at 0.2 m/s.
    assert 0.8 < T[-1] < 1.0 and 0.08 < steps[-1] < 0.10
    assert _period(gen, style, (0.0, 0.0, 0.0)) == pytest.approx(style.base["period_max"])
    fast = style.copy()
    fast.base["period_min"] = 1.1
    assert _period(gen, fast, (0.2, 0.0, 0.0)) == pytest.approx(1.1)


@pytest.mark.parametrize("cmd", [(0.1, 0.05, 0.3), (-0.15, 0.1, -0.8), (0.0, 0.15, 1.2)])
def test_cadence_is_mirror_symmetric(gen, cmd):
    style = Style()
    assert _period(gen, style, cmd) == _period(gen, style, (cmd[0], -cmd[1], -cmd[2]))


def test_turning_caps_yaw_per_step(gen):
    style = Style()
    for wz in (0.3, 0.6, 0.9, 1.2):
        T = _period(gen, style, (0.0, 0.0, wz))
        assert wz * T / 2 <= math.radians(style.base["max_step_yaw"]) + 1e-9


def test_legacy_fixed_period_style(gen):
    """A style saved with the old global `period` keeps exactly that cadence everywhere."""
    style = Style.from_dict({"name": "old", "base": {"period": 0.84, "heel_strike": 5.0}})
    for cmd in [(0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.2, 0.1, 0.0), (0.0, 0.0, 1.2)]:
        assert _period(gen, style, cmd) == pytest.approx(0.84)


def test_cycle_is_phase_normalised(gen):
    c = gen.cycle(Style(), (0.15, 0.0, 0.0))
    assert c.n_frames == gen.r.n_phase
    assert c.period == pytest.approx(_period(gen, Style(), (0.15, 0.0, 0.0)))
    assert c.dt == pytest.approx(c.period / c.n_frames)


# ---------------------------------------------------------------------- rocker model
def _profile(gen, style, cmd, side="right"):
    c = gen.cycle(style, cmd)
    u, p = c.pitch_profiles[side]
    return c, u, np.degrees(p)        # + = toes down


def test_heel_strike_grows_with_step_length(gen):
    style = Style()
    strikes = [-_profile(gen, style, (v, 0.0, 0.0))[2][0] for v in (0.05, 0.1, 0.2)]
    assert strikes[0] > 2.0 and all(a < b for a, b in zip(strikes, strikes[1:]))
    _, _, p = _profile(gen, style, (0.2, 0.0, 0.0))
    assert p[-1] > 1.0                      # and the heel lifts before toe off
    style.base["heel_off_angle"] = 5.0      # a smaller ankle range lifts it sooner and higher
    _, _, p5 = _profile(gen, style, (0.2, 0.0, 0.0))
    assert p5[-1] > p[-1] + 3.0


def test_backward_walking_reverses_the_rockers(gen):
    _, _, p = _profile(gen, Style(), (-0.2, 0.0, 0.0))
    assert p[0] > 3.0                       # toe strike
    assert p[-1] < -1.0                     # heel last


@pytest.mark.parametrize("cmd", [(0.0, 0.15, 0.0), (0.0, -0.1, 0.0), (0.0, 0.0, 0.0)])
def test_no_sagittal_rocker_in_place_or_sideways(gen, cmd):
    """No forward leg rotation over the foot -> (almost) no heel/toe. Stepping in place may land
    slightly toe first, as humans march."""
    _, _, p = _profile(gen, Style(), cmd)
    limit = 4.0 if cmd == (0.0, 0.0, 0.0) else 2.0
    assert np.abs(p).max() < limit


def test_rollover_gain_zero_is_flat(gen):
    style = Style()
    style.base["rollover_gain"] = 0.0
    c = gen.cycle(style, (0.2, 0.0, 0.0))
    assert c.pitch_profiles is None


def test_contact_ankle_sets_the_landing(gen):
    """`contact_ankle` is the ankle at touchdown: +5 deg of it lands the foot 5 deg more toes-up."""
    style = Style()
    base = _profile(gen, style, (0.1, 0.0, 0.0))[2][0]
    style.base["contact_ankle"] = 5.0
    more = _profile(gen, style, (0.1, 0.0, 0.0))[2][0]
    assert more - base == pytest.approx(-5.0, abs=0.75)


def test_ik_tracks_the_landing_pitch(gen):
    """The right sole's pitch at phase 0 (right touchdown) is the profile's landing pitch."""
    c, _, p = _profile(gen, Style(), (0.2, 0.0, 0.0))
    gen.r.forward(c.qpos[0])
    R = gen.r.d.site_xmat[gen.r.sole_site_ids["right"]].reshape(3, 3)
    pitch = math.degrees(math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0])))   # + = toes down
    assert pitch == pytest.approx(p[0], abs=0.5)


# ---------------------------------------------------------------------- direction-aware knobs
def test_twist_follows_forward_step_length(gen):
    style = Style()
    style.base["twist"] = 6.0
    st = style.resolve((0.0, 0.0, 0.0), gen.r.command_ranges)
    assert gen._twist_amplitude((0.0, 0.0, 0.0), st, 2.0) == 0.0
    assert gen._twist_amplitude((0.0, 0.15, 0.0), st, 0.7) == 0.0
    T = _period(gen, style, (0.2, 0.0, 0.0))
    assert gen._twist_amplitude((0.2, 0.0, 0.0), st, T) == pytest.approx(math.radians(6.0))


def test_positive_twist_puts_the_landing_hip_forward(gen):
    """+ twist: at phase 0 (right heel strike) the right hip is ahead of the left (in-phase)."""
    def hip_lead(twist):
        style = Style()
        style.base["twist"] = twist
        c = gen.cycle(style, (0.2, 0.0, 0.0))
        gen.r.forward(c.qpos[0])
        ids = gen.r.leg_joint_ids
        heading = c.path[0, 2]
        fwd = np.array([math.cos(heading), math.sin(heading), 0.0])
        return float((gen.r.d.xanchor[ids["right"]["hip_pitch"]] - gen.r.d.xanchor[ids["left"]["hip_pitch"]]) @ fwd)

    assert hip_lead(6.0) - hip_lead(0.0) > 0.003


def test_side_steps_lift_less_and_the_closing_step_least(gen):
    """Stepping left: the left foot leads (abducts), the right one closes. Both lift less than a
    forward step, the closing one 1/1.5 of the leading one (Cappellini et al. 2023)."""
    style = Style()
    c = gen.cycle(style, (0.0, 0.15, 0.0))
    fh = c.field("foot_height")
    contact = c.field("contact")
    lead = fh[contact[:, 0] == 0, 0].max()
    close = fh[contact[:, 1] == 0, 1].max()
    h = style.base["step_height"]
    assert lead < 0.8 * h and close < lead
    assert lead / close == pytest.approx(1.5, rel=0.15)
    fwd = gen.cycle(style, (0.1, 0.0, 0.0)).field("foot_height").max()
    assert fwd > 0.9 * h
