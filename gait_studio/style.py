"""Gait style parameters and their blending across the command space.

A Style is a set of *base* knob values (the step-in-place gait) plus optional per-anchor
overrides. Anchors sit at the extremes of the command space (max forward, max backward, max
sideways, max turning). For a command, every anchor contributes its delta from the base,
weighted linearly by how far the command reaches toward that anchor. This is a simplified
take on Disney's blending of authored walk cycles (Hoffman et al., SIGGRAPH 2024).

Signed knobs (leans) are mirrored by the sign of the motion that drives them, so a lean set
on the "sideways" anchor leans left when stepping left and right when stepping right. Their
base value is fixed at zero, otherwise the step-in-place gait would not be symmetric.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field

import yaml


@dataclass(frozen=True)
class Knob:
    name: str
    label: str
    group: str
    default: float
    min: float
    max: float
    step: float
    unit: str = ""
    anchorable: bool = True
    signed: bool = False
    help: str = ""


KNOBS: list[Knob] = [
    # --- Timing. Cadence follows the human speed law (cadence.py); none of these blend by anchor
    # because the law already depends on the command.
    Knob("cadence_scale", "Cadence scale", "Timing", 1.0, 0.6, 1.6, 0.02, "x", anchorable=False,
         help="Multiplies the human (Froude-scaled) step frequency. >1 = quicker, shorter steps."),
    Knob("cadence_exponent", "Speed → cadence", "Timing", 0.5, 0.0, 0.8, 0.02, "", anchorable=False,
         help="Cadence ∝ speed^this (humans ~0.42-0.58). 0 = constant cadence, all speed from step length."),
    Knob("period_max", "Slowest cycle", "Timing", 2.0, 0.36, 2.4, 0.02, "s", anchorable=False,
         help="Cap on the cycle at slow speeds. The human law gives ~1.9 s at 0.05 m/s for this leg."),
    Knob("period_min", "Fastest cycle", "Timing", 0.5, 0.36, 1.2, 0.02, "s", anchorable=False,
         help="Floor on the cycle at high speeds."),
    Knob("max_step_yaw", "Max turn per step", "Timing", 25.0, 10.0, 40.0, 0.5, "deg", anchorable=False,
         help="Quickens the cadence when turning so one step never turns more than this (hip yaw limit 45°)."),
    Knob("double_support", "Double support", "Timing", 0.10, 0.03, 0.50, 0.01, "frac",
         help="Fraction of each step with both feet on the ground."),
    # --- Feet
    Knob("stance_width", "Stance width", "Feet", 0.060, 0.048, 0.160, 0.002, "m",
         help="Lateral distance between sole centres. 0.06 puts the feet under the hips (hip-pitch axes "
              "6.0 cm apart); wider splays the legs and needs proportionally more sway."),
    Knob("min_foot_gap", "Min foot gap", "Feet", 0.012, 0.000, 0.040, 0.002, "m", anchorable=False,
         help="Clearance kept between the inner sole edges; widens the stance for sidesteps."),
    Knob("toe_out", "Toe-out", "Feet", 0.0, -10.0, 20.0, 0.5, "deg",
         help="Foot yaw outward."),
    Knob("step_height", "Step height", "Feet", 0.010, 0.005, 0.060, 0.001, "m",
         help="Swing-foot apex above the ground."),
    Knob("swing_ease", "Swing ease", "Feet", 0.20, 0.05, 0.50, 0.01, "frac",
         help="Fraction of the swing spent speeding up / slowing down. Lower = lower peak joint speed, harsher start/stop."),
    Knob("apex_timing", "Apex timing", "Feet", 0.50, 0.25, 0.75, 0.01, "frac",
         help="When in the swing the apex happens (0.5 = middle)."),
    Knob("swing_overshoot", "Reach overshoot", "Feet", 0.0, -0.30, 0.50, 0.01, "frac",
         help="Swing foot reaches past its target and settles back (fraction of the step)."),
    Knob("swing_bulge", "Swing bulge", "Feet", 0.0, -0.010, 0.030, 0.001, "m",
         help="Outward arc of the swing foot."),
    Knob("lateral_lift", "Side-step lift", "Feet", 0.6, 0.3, 1.0, 0.02, "x",
         help="Foot lift for sideways steps, relative to step height (humans side-step with a nearly "
              "straight knee). The closing step lifts 1/1.5 of the leading one."),
    # --- Heel-to-toe: the rocker model (rollover.py). Foot pitch follows the shank, not a schedule.
    Knob("rollover_gain", "Heel-to-toe amount", "Feet", 1.0, 0.0, 1.5, 0.05, "x",
         help="Scales the heel strike / toe off that the rocker model derives from leg geometry. 0 = flat feet."),
    Knob("contact_ankle", "Ankle at contact", "Feet", 0.0, -10.0, 15.0, 0.5, "deg",
         help="Ankle dorsiflexion at touchdown, relative to mid-stance (hip over the ankle; humans ~0). "
              "The foot lands pitched by the shank's lean plus this, so heel strike grows with step length."),
    Knob("heel_roll_time", "Loading response", "Feet", 0.17, 0.05, 0.40, 0.01, "frac",
         help="Fraction of the stance spent lowering the foot from heel (or toe) contact to flat (humans ~0.17)."),
    Knob("heel_off_angle", "Heel-off ankle angle", "Feet", 10.0, 2.0, 30.0, 0.5, "deg",
         help="How far the shank may lean over the flat foot, either way from mid-stance, before the foot "
              "pivots onto an edge: the heel lifts walking forward (humans ~10)."),
    Knob("max_leg_reach", "Max leg reach", "Feet", 0.97, 0.85, 1.0, 0.005, "frac", anchorable=False,
         help="Fraction of the straight hip-ankle length. Beyond it the stance foot pivots on its edge to keep reaching."),
    Knob("push_off", "Push-off", "Feet", 0.0, 0.0, 20.0, 0.5, "deg",
         help="Extra plantarflexion through the final double support (humans ~15-20). The ZMP must be at the toe then."),
    # --- Body
    Knob("com_height", "CoM height", "Body", 0.195, 0.150, 0.220, 0.001, "m",
         help="Lower = more knee bend."),
    Knob("com_forward", "CoM forward offset", "Body", 0.0, -0.020, 0.020, 0.001, "m",
         help="Shifts the CoM ahead of the feet."),
    Knob("sway_gain", "Lateral sway gain", "Body", 1.0, 0.3, 2.0, 0.05, "x",
         help="Scales the LIPM side-to-side CoM motion. 1 = dynamically consistent; below 1 the ZMP leaves "
              "the foot (watch ZMP margin). To sway less, use ZMP inward and a narrower stance instead."),
    Knob("zmp_inward", "ZMP inward", "Body", 0.6, 0.0, 1.0, 0.05, "frac",
         help="Where the ZMP sits across the stance foot: 0 = sole centre, 1 = inner edge (5 mm margin). "
              "Sway is proportional to the ZMP's distance from the midline, so inward = less sway."),
    Knob("bob_amplitude", "Vertical bob", "Body", 0.0, 0.0, 0.015, 0.0005, "m",
         help="CoM up/down oscillation, twice per cycle."),
    Knob("bob_phase", "Bob phase", "Body", 0.0, -180.0, 180.0, 5.0, "deg"),
    Knob("trunk_pitch", "Trunk pitch", "Body", 0.0, -15.0, 20.0, 0.5, "deg",
         help="Forward lean of the torso."),
    Knob("trunk_roll_sway", "Roll sway", "Body", 0.0, -8.0, 8.0, 0.25, "deg",
         help="Torso roll toward the stance foot."),
    Knob("twist", "Pelvis twist", "Body", 0.0, -12.0, 12.0, 0.25, "deg",
         help="Pelvis yaw at max forward speed, scaled by the forward step length (none in place or sideways). "
              "+ = the swing-side hip leads at heel strike; humans this slow rotate the other way (-)."),
    Knob("trunk_lean", "Lean into motion", "Body", 0.0, -15.0, 15.0, 0.5, "deg", signed=True,
         help="Torso roll into sidesteps / turns. Set on the Sideways or Turning anchor."),
]

KNOB_BY_NAME = {k.name: k for k in KNOBS}
GROUPS = ["Timing", "Feet", "Body"]


@dataclass(frozen=True)
class Anchor:
    name: str
    label: str
    axis: str        # "vx", "vy", "wz"
    direction: int   # +1 / -1 for vx; vy/wz anchors are mirrored (|value|)


ANCHORS: list[Anchor] = [
    Anchor("forward", "Forward (max vx)", "vx", +1),
    Anchor("backward", "Backward (min vx)", "vx", -1),
    Anchor("sideways", "Sideways (max |vy|)", "vy", +1),
    Anchor("turning", "Turning (max |wz|)", "wz", +1),
]
ANCHOR_BY_NAME = {a.name: a for a in ANCHORS}


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# Out-of-the-box anchor overrides. "Reset all" returns to these. Heel-to-toe needs none: the rocker
# model scales it with the step. Side steps hike the swing hip (roll toward the stance foot), the
# clearance humans use instead of knee bend.
DEFAULT_ANCHORS = {
    "sideways": {"trunk_roll_sway": 2.0},
}

# Knobs that no longer exist. heel_strike / toe_off / toe_roll_time were hand-set angles, replaced
# by the rocker model; a fixed `period` became the cadence law.
LEGACY_KNOBS = ("period", "heel_strike", "toe_off", "toe_roll_time")


@dataclass
class Style:
    name: str = "default"
    base: dict = field(default_factory=lambda: {k.name: k.default for k in KNOBS})
    anchors: dict = field(default_factory=lambda: {a.name: dict(DEFAULT_ANCHORS.get(a.name, {})) for a in ANCHORS})

    def __post_init__(self):
        for k in KNOBS:
            self.base.setdefault(k.name, k.default)
            if k.signed:
                self.base[k.name] = 0.0
        for a in ANCHORS:
            self.anchors.setdefault(a.name, {})

    # ------------------------------------------------------------------ blending
    def anchor_weights(self, cmd, ranges) -> dict[str, tuple[float, float]]:
        """Return {anchor: (weight, sign)} for a command (vx, vy, wz)."""
        vx, vy, wz = cmd
        vx_lo, vx_hi = ranges["vx"]
        vy_hi = max(abs(ranges["vy"][0]), abs(ranges["vy"][1]))
        wz_hi = max(abs(ranges["wz"][0]), abs(ranges["wz"][1]))
        w = {
            "forward": (_clamp(vx / vx_hi, 0.0, 1.0) if vx_hi > 0 else 0.0, 1.0),
            "backward": (_clamp(vx / vx_lo, 0.0, 1.0) if vx_lo < 0 else 0.0, 1.0),
            "sideways": (_clamp(abs(vy) / vy_hi, 0.0, 1.0) if vy_hi > 0 else 0.0, 1.0 if vy >= 0 else -1.0),
            "turning": (_clamp(abs(wz) / wz_hi, 0.0, 1.0) if wz_hi > 0 else 0.0, 1.0 if wz >= 0 else -1.0),
        }
        return w

    def resolve(self, cmd, ranges) -> dict[str, float]:
        """Knob values for a given command."""
        out = dict(self.base)
        weights = self.anchor_weights(cmd, ranges)
        for aname, overrides in self.anchors.items():
            wgt, sgn = weights[aname]
            if wgt <= 0.0:
                continue
            for kname, val in overrides.items():
                knob = KNOB_BY_NAME.get(kname)
                if knob is None or not knob.anchorable:
                    continue
                delta = val - self.base[kname]
                out[kname] += wgt * delta * (sgn if knob.signed else 1.0)
        for k in KNOBS:
            lo, hi = (-max(abs(k.min), abs(k.max)), max(abs(k.min), abs(k.max))) if k.signed else (k.min, k.max)
            out[k.name] = _clamp(out[k.name], lo, hi)
        return out

    # ------------------------------------------------------------------ editing
    def get(self, knob: str, anchor: str | None = None) -> float:
        if anchor is None:
            return self.base[knob]
        return self.anchors[anchor].get(knob, self.base[knob])

    def set(self, knob: str, value: float, anchor: str | None = None) -> None:
        k = KNOB_BY_NAME[knob]
        if anchor is None:
            if k.signed:
                return  # signed knobs live on anchors only
            self.base[knob] = float(value)
        else:
            if not k.anchorable:
                self.base[knob] = float(value)
                return
            self.anchors[anchor][knob] = float(value)

    def clear_override(self, knob: str, anchor: str) -> None:
        self.anchors[anchor].pop(knob, None)

    def is_overridden(self, knob: str, anchor: str) -> bool:
        return knob in self.anchors[anchor]

    def is_modified(self, knob: str, anchor: str | None = None) -> bool:
        """Whether the slider for this knob (at this anchor) differs from a fresh reset."""
        k = KNOB_BY_NAME[knob]
        if anchor is None or not k.anchorable:
            return abs(self.base[knob] - (0.0 if k.signed else k.default)) > 1e-9
        default = DEFAULT_ANCHORS.get(anchor, {}).get(knob)
        current = self.anchors[anchor].get(knob)
        if default is None or current is None:
            return default is not current
        return abs(current - default) > 1e-9

    def reset_knob(self, knob: str, anchor: str | None = None) -> None:
        """Back to what a fresh Style() has: the knob default on the base, the default override
        (or none, so the base blends in) on an anchor."""
        k = KNOB_BY_NAME[knob]
        if anchor is None or not k.anchorable:
            self.base[knob] = 0.0 if k.signed else k.default
            return
        default = DEFAULT_ANCHORS.get(anchor, {}).get(knob)
        if default is None:
            self.anchors[anchor].pop(knob, None)
        else:
            self.anchors[anchor][knob] = default

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {"name": self.name, "base": dict(self.base),
                "anchors": {k: dict(v) for k, v in self.anchors.items() if v}}

    @classmethod
    def from_dict(cls, d: dict) -> "Style":
        base = {k.name: k.default for k in KNOBS}
        raw = d.get("base") or {}
        base.update({k: float(v) for k, v in raw.items() if k in KNOB_BY_NAME})
        if "period" in raw and not any(k in raw for k in ("period_min", "period_max")):
            # A fixed-period style keeps its cadence exactly: the law clamps to [period, period].
            base["period_min"] = base["period_max"] = float(raw["period"])
        dropped = sorted({k for k in raw if k in LEGACY_KNOBS and k != "period"}
                         | {k for ov in (d.get("anchors") or {}).values() for k in (ov or {}) if k in LEGACY_KNOBS})
        if dropped:
            print(f"style '{d.get('name', 'style')}': ignoring legacy knob(s) {dropped} "
                  f"(heel-to-toe now comes from the rocker model)")
        anchors = {a.name: {} for a in ANCHORS}
        for aname, ov in (d.get("anchors") or {}).items():
            if aname in anchors:
                anchors[aname] = {k: float(v) for k, v in ov.items() if k in KNOB_BY_NAME}
        return cls(name=d.get("name", "style"), base=base, anchors=anchors)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "Style":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def copy(self) -> "Style":
        return copy.deepcopy(self)

    def digest(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:10]
