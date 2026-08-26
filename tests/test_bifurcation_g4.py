"""Gate G4 of the bifurcation pre-registration: the bent-pipe control.

docs/bifurcation-preregistration.md, order of work, step 1: before the
junction measures anything new, the junction MACHINERY — composite arc
geometry, region-wise analytic velocity, path-based receiver windows,
capped ends — must reproduce the straight-pipe record it descends from.

THE INSTABILITY THIS GATE CAUGHT (2026-08-26, the reason G4 runs first):
stock BGK ADE at the stability-pinned tau is unstable whenever advection
runs oblique to the lattice — a straight pipe tilted 30 degrees diverges
with no bend at all. The full discriminating ladder is in the
pre-registration addendum and the record; the two structural facts are
re-derived on every run below rather than quoted: (1) the uniform scheme
is plane-wave stable at the operating point, so the wall interaction is
the mechanism (oblique flow drives advective flux through the staircase's
bounce-back faces; the axis-aligned case has exactly none, which is why
the straight programme never saw it); (2) the TRT's ODD rate carries the
diffusivity, so running TRT at magic Lambda = 1/4 — the configuration the
ladder found stable — changes stability only, never the physical D.

What this test runs TODAY: the two angle-0 legs.
  A0  BGK — the MACHINERY gate: same collision as the straight record,
      so the pre-registered envelope (peaks 0.4%, tail ratios 0.04)
      isolates the new geometry/readout code path.
  A1  TRT Lambda = 1/4 — the collision-change measurement, recorded
      against the record with the mechanism re-derived in-test: at
      finite wavenumber the two collisions' effective diffusivities
      separate (BGK grows with k, TRT shrinks — ratio 0.83 at k = 1),
      so TRT resolves sharper peaks; its k -> 0 physics is identical
      (structural fact 2). Tails and mass must still hold the envelope.
The bend leg is BLOCKED, and that is
the record's finding, not a gap: TRT Lambda = 1/4 reduces the wall-loop
gain by orders yet the drain persists at production resolution (res 12,
angle 30: bulk mass crosses ZERO at 3.27 s with clean-looking peaks —
res 6's apparent stability was a rate effect, fewer steps per physical
time). Stable oblique scalar transport needs a NO-FLUX interpolated wall
for the ADE lattice; stock OpenLB 1.9 ships only a Bouzidi-ADE-Dirichlet
(absorbing) wall, so that scheme is the named next rung before any bend
or junction measurement.
"""

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from betaflow.analytic import lattice_boltzmann as lb
from betaflow.provenance import git_sha
from betaflow.runners import run_case

REPO = Path(__file__).resolve().parents[1]
OPENLB_ROOT = Path.home() / "GitHub" / "openlb"
CASE_FILE = REPO / "betaflow" / "cases" / "mc_channel.yaml"
STRAIGHT_RECORD = REPO / "results" / "mc_channel_openlb.json"
RESULTS_FILE = REPO / "results" / "bifurcation_g4_control.json"

pytestmark = pytest.mark.slow

# The pre-registered G4 envelope (docs/bifurcation-preregistration.md).
PEAK_ENVELOPE = 0.004
TAIL_ENVELOPE = 0.04

_C = np.array(lb.VELOCITY_SETS["D3Q7"][0], float)
_W = np.array(lb.VELOCITY_SETS["D3Q7"][1], float)
_CS2 = 0.25
_OPP = [0, 2, 1, 4, 3, 6, 5]


def _bgk_spectral_radius(tau, u, nk=12):
    """max |eigenvalue| of the exact uniform-BGK amplification matrix over
    a 3-D wavevector grid."""
    q = _W * (1.0 + (_C @ np.asarray(u, float)) / _CS2)
    worst = 0.0
    ks = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    for kv in itertools.product(ks, ks, ks):
        if kv == (0.0, 0.0, 0.0):
            continue
        ph = np.exp(-1j * (_C @ np.array(kv)))
        G = ph[:, None] * ((1.0 - 1.0/tau) * np.eye(7) + q[:, None] / tau)
        worst = max(worst, float(np.abs(np.linalg.eigvals(G)).max()))
    return worst


def _trt_d_eff(u, om_even, om_odd, k=1e-3):
    """D_eff of OpenLB's ADE TRT from its exact amplification matrix —
    the check that the diffusivity rides the ODD rate."""
    q = _W * (1.0 + (_C @ np.asarray(u, float)) / _CS2)
    P_plus = np.zeros((7, 7))
    P_minus = np.zeros((7, 7))
    for i in range(7):
        P_plus[i, i] += 0.5
        P_plus[i, _OPP[i]] += 0.5
        P_minus[i, i] += 0.5
        P_minus[i, _OPP[i]] -= 0.5
    Qe = np.outer(q, np.ones(7))
    M = (np.eye(7) - om_even * (P_plus - P_plus @ Qe)
                   - om_odd * (P_minus - P_minus @ Qe))
    ph = np.exp(-1j * (_C @ np.array([k, 0.0, 0.0])))
    lam = np.linalg.eigvals(ph[:, None] * M)
    lam = lam[np.argmax(np.abs(lam))]
    return float(-np.log(lam).real / k**2)


def _receiver_metrics(res):
    """Exactly the metrics the straight record persists, same formulas."""
    out = []
    for rec in res["receivers"]:
        t, cm = rec["t"], rec["cir_measured"]
        co = rec["cir_reference_straight"]
        rsel = (t >= 3.0 * rec["t2"]) & (t <= 6.0 * rec["t2"])
        out.append({
            "dbar_um": rec["dbar"] * 1e6,
            "peak_measured": float(np.max(cm)),
            "peak_lag_relative": float(t[np.argmax(cm)]) / rec["t2"] - 1.0,
            "ringing_min": float(np.min(cm)),
            "tail_ratio_3_to_6_t2": float(np.mean(cm[rsel])
                                          / np.mean(co[rsel])),
        })
    return out


@pytest.mark.skipif(
    not (OPENLB_ROOT / "build" / "lib" / "libolbcore.a").is_file(),
    reason="OpenLB build tree not present")
def test_bifurcation_g4_control():
    case = yaml.safe_load(CASE_FILE.read_text())
    straight = json.loads(STRAIGHT_RECORD.read_text())["receivers"]

    # Structural fact 1: the uniform scheme is plane-wave stable at the
    # pinned tau for oblique flow, so the measured BGK divergence was the
    # wall interaction, and the TRT choice below aims at the mechanism.
    tau_pinned = 0.5048006542216479
    u30 = (0.04 * np.cos(np.pi/6), 0.04 * np.sin(np.pi/6), 0.0)
    rho = _bgk_spectral_radius(tau_pinned, u30)
    assert rho < 1.0 + 1e-12, (
        f"plane-wave scan now UNSTABLE ({rho}); the wall attribution in "
        f"this test's docstring no longer holds — re-diagnose")

    # Structural fact 2: D rides the TRT's odd rate — pinning the odd rate
    # and moving the even one must leave D_eff unchanged, or Lambda = 1/4
    # would be silently changing the physics.
    om_odd = 1.0 / tau_pinned
    d_ref = _trt_d_eff((0.04, 0, 0), om_odd, om_odd)
    for tau_even in (1.0, 52.583):
        d = _trt_d_eff((0.04, 0, 0), 1.0 / tau_even, om_odd)
        # Tolerance origin: the extraction reads -Re ln(lambda)/k^2 at
        # k = 1e-3, whose O(k^2) truncation term is even-rate-DEPENDENT at
        # relative ~1e-6 (that dependence at finite k is structural fact 3
        # below). The k -> 0 claim is gated an order above that floor —
        # against the 17% finite-k separation it discriminates from.
        assert abs(d / d_ref - 1.0) < 1e-5, (
            f"TRT D_eff moved with the even rate (tau_even={tau_even}): "
            f"{d} vs {d_ref} — the Lambda choice is not physics-neutral")

    # Structural fact 3, the peak mechanism: the finite-k diffusivities
    # separate between the collisions (BGK up, TRT down with k) while the
    # k -> 0 values agree — re-derived so the recorded peak differences
    # always sit next to their mechanism.
    om_even_magic = 1.0 / (0.25 / (tau_pinned - 0.5) + 0.5)
    d0_b = _trt_d_eff((0.04, 0, 0), om_odd, om_odd, k=1e-3)
    d0_t = _trt_d_eff((0.04, 0, 0), om_even_magic, om_odd, k=1e-3)
    d1_b = _trt_d_eff((0.04, 0, 0), om_odd, om_odd, k=1.0)
    d1_t = _trt_d_eff((0.04, 0, 0), om_even_magic, om_odd, k=1.0)
    # same k = 1e-3 extraction floor as structural fact 2 above
    assert abs(d0_t / d0_b - 1.0) < 1e-5
    finite_k_ratio = d1_t / d1_b
    assert finite_k_ratio < 0.9, (
        f"the finite-k separation ({finite_k_ratio:.3f} at k=1) no longer "
        f"explains a TRT peak elevation — re-diagnose before quoting one")

    legs = {}
    for tag, dyn in (("A0", "bgk"), ("A1", "trt")):
        res = run_case(case, runner="openlb", bend_angle_deg=0.0,
                       dynamics=dyn, magic_lambda=0.25)
        m = res["mass_over_initial"]
        assert m[0] == pytest.approx(1.0)
        # G1 bookkeeping: parking only, bounded, never growth.
        assert m[-1] > 0.94, (
            f"leg {tag}: bulk mass fell to {m[-1]:.4f} of initial")
        assert m[-1] < 1.005, (
            f"leg {tag}: bulk mass GREW to {m[-1]:.4f} — a leak inward")
        legs[tag] = {
            "angle_deg": 0.0,
            "dynamics": dyn,
            "receivers": _receiver_metrics(res),
            "mass_final_over_initial": float(m[-1]),
            "meta": res["meta"],
        }

    # THE MACHINERY GATE (leg A0, BGK): the pre-registered envelope
    # against the committed straight prescribed record — same collision,
    # so only the new code path is on trial.
    control_deviation = []
    for ctrl, ref in zip(legs["A0"]["receivers"], straight):
        assert ctrl["dbar_um"] == pytest.approx(ref["dbar_um"])
        dpeak = ctrl["peak_measured"] / ref["peak_measured"] - 1.0
        dtail = ctrl["tail_ratio_3_to_6_t2"] - ref["tail_ratio_3_to_6_t2"]
        assert abs(dpeak) < PEAK_ENVELOPE, (
            f"G4 machinery, dbar={ctrl['dbar_um']:.0f}um: peak deviates "
            f"{dpeak:+.2%} from the straight record — diagnose before any "
            f"junction run")
        assert abs(dtail) < TAIL_ENVELOPE, (
            f"G4 machinery, dbar={ctrl['dbar_um']:.0f}um: tail ratio "
            f"deviates {dtail:+.3f} from the straight record")
        control_deviation.append({
            "dbar_um": ctrl["dbar_um"],
            "peak_relative_to_straight_record": dpeak,
            "tail_ratio_minus_straight_record": dtail,
        })

    # THE COLLISION MEASUREMENT (leg A1, TRT): peaks may sit above the
    # BGK record by the finite-k mechanism; tails must hold the envelope.
    collision_change = []
    for ctrl, ref in zip(legs["A1"]["receivers"], straight):
        dtail = ctrl["tail_ratio_3_to_6_t2"] - ref["tail_ratio_3_to_6_t2"]
        assert abs(dtail) < TAIL_ENVELOPE, (
            f"TRT leg, dbar={ctrl['dbar_um']:.0f}um: tail ratio deviates "
            f"{dtail:+.3f} — the collision change should sharpen peaks, "
            f"never move tails beyond the envelope")
        collision_change.append({
            "dbar_um": ctrl["dbar_um"],
            "peak_relative_to_straight_record":
                ctrl["peak_measured"] / ref["peak_measured"] - 1.0,
            "tail_ratio_minus_straight_record": dtail,
        })

    record = {
        "case": "mc_channel through the bent-pipe G4 control",
        "preregistration": "docs/bifurcation-preregistration.md, gate G4 "
                           "and the oblique-wall addendum",
        "oblique_wall_instability": {
            "finding": "stock BGK ADE at the stability-pinned tau is "
                       "unstable for advection oblique to the lattice; a "
                       "straight pipe tilted 30 degrees diverges with no "
                       "bend present",
            "ladder": [
                "mitre BGK tau=0.5048: mass -1.9e51",
                "mitre BGK tau=0.5096: mass -5.8e46 (damping not the knob)",
                "smooth-arc BGK: diverges (kink not the mechanism)",
                "straight oblique BGK: diverges (wall IS the mechanism)",
                "straight oblique BGK 2nd-order eq: diverges",
                "oblique TRT tau_even=1: diverges (even sector not the knob)",
                "oblique TRT Lambda=1/4, res 6: no blowup over the horizon "
                "(mass 0.67)",
                "oblique TRT Lambda=1/4, res 12: mass crosses ZERO at "
                "3.27 s of 3.44 - the gain is reduced by orders, the loop "
                "persists; res 6 apparent stability was a rate effect",
            ],
            "plane_wave_max_lambda_at_pin": rho,
            "resolution_chosen": "TRT, magic Lambda = 1/4 (Ginzburg's "
                                 "bounce-back stability optimum); D rides "
                                 "the odd rate, checked above, so the "
                                 "physics is untouched",
            "bend_leg_status": "BLOCKED pending a no-flux interpolated "
                               "ADE wall: stock OpenLB 1.9 ships only "
                               "Bouzidi-ADE-Dirichlet (absorbing), and "
                               "impermeable walls are the channel's "
                               "physics. The wall scheme is the named "
                               "next rung; no bend number is quoted "
                               "before it lands.",
        },
        "legs": {k: {kk: vv for kk, vv in v.items() if kk != "meta"}
                 for k, v in legs.items()},
        "machinery_gate_bgk_vs_straight_record": control_deviation,
        "collision_change_trt_vs_straight_record": {
            "receivers": collision_change,
            "mechanism": "finite-k diffusivity separation, re-derived "
                         "in-test: D_eff(k=1) TRT/BGK = "
                         f"{finite_k_ratio:.3f} with identical k->0 "
                         "values; BGK smooths high-k content harder, so "
                         "TRT resolves sharper peaks at 12 cells/radius "
                         "(the BGK record itself sits ~+15% over the "
                         "slug-averaged analytic reference at the peak)",
        },
        "control_envelope": {"peak": PEAK_ENVELOPE, "tail": TAIL_ENVELOPE,
                             "origin": "the straight case's coupled-vs-"
                                       "prescribed cost envelope, as "
                                       "pre-registered"},
        "slug_flakiness_found_by_this_gate": {
            "finding": "the straight app's slug boundary |x-x0| <= dx "
                       "falls exactly on cell centres; rounding includes "
                       "3 slices there (initial total 1317) where this "
                       "app's different x0 arithmetic included 2 (878) - "
                       "a 50% release-width difference that surfaced as "
                       "a +1.5-4% peak discrepancy in the first "
                       "machinery comparison",
            "fix_here": "deterministic 3-slice slug (half-width 1.01 dx),"
                        " realised width printed as slugW",
            "flag_for_straight_record": "its provenance slugW = 2 dx but "
                                        "the realised slug is 3 slices "
                                        "(50 um); its slug-averaged "
                                        "reference is therefore computed "
                                        "slightly narrow - refresh on the "
                                        "record's next regeneration",
        },
        "mass_final_over_initial": {
            k: v["mass_final_over_initial"] for k, v in legs.items()},
        "meta": {k: v["meta"] for k, v in legs.items()},
        "git_sha": git_sha(REPO),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")
