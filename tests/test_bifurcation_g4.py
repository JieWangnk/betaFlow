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
THE WALL THAT WORKS (2026-08-27): Noble-Torczynski partially-saturated
cells (solid fraction from the true surface, pairwise-antisymmetric solid
operator — exactly conservative) with a TRT bulk at magic Lambda = 1/4.
Either ingredient alone fails: NT-BGK diverges obliquely, TRT+bounce-back
drains to zero. Together they are stable at the pinned tau, production
resolution, oblique flow. Two more legs run on that scheme:
  N0   angle 0,  NT+TRT — the scheme's own control: its deltas against
       the straight record ARE the new instrument's calibration (TRT
       sharpens peaks; the near-wall tail content sits partly in cut
       cells under bulk-only accounting, so tail ratios read low). No
       envelope gate — the envelope belongs to A0, same-instrument.
  N30  angle 30, NT+TRT — THE FIRST BEND MEASUREMENT, reported against
       N0 (same instrument, one variable). Gated only on its in-run
       control: window 1 lies upstream of the bend and must match N0.
       The wall-layer budget of the reference-lattice study (amp ~
       5 u_lat over 2-3 cells) rides with the numbers.

THE SOLVED-FLOW LEG (2026-09-01): S30 — the same bend with the flow
SOLVED by bentFlow3d (D3Q19, Bouzidi walls, inlet Poiseuille, outlet
pressure, warm-started from the analytic field) instead of prescribed,
the same staging order the straight coupled model used. The fluid stage
is gated on the pre-registered G2/G7 values (flux imbalance 5e-3 —
measured 7.8e-4; profile L2 2e-2 at both stations — measured ~5e-3) and
on P2 (no recirculation: the bend-centreline minimum tangential velocity
must exceed half the centreline speed — measured 0.998 of it). The
field hand-off carries a hard guard in the scalar app: a half-cell
grid-convention mismatch once made every lookup miss and the scalar
crawled on a silently-zero field (lag +51 t2); the guard now aborts
unless the loaded field carries the centreline speed at the slug.
S30 vs N30 is the solved-vs-prescribed cost ON THE BEND, the analogue
of the straight coupled model's 0.1-0.3% peak cost.
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
    fluid_stage = None
    for tag, angle, dyn, wall, vel in (
            ("A0", 0.0, "bgk", "bb", "analytic"),
            ("A1", 0.0, "trt", "bb", "analytic"),
            ("N0", 0.0, "trt", "nt", "analytic"),
            ("N30", 30.0, "trt", "nt", "analytic"),
            ("S30", 30.0, "trt", "nt", "solved")):
        res = run_case(case, runner="openlb", bend_angle_deg=angle,
                       dynamics=dyn, magic_lambda=0.25, wall=wall,
                       velocity=vel)
        if vel == "solved":
            fluid_stage = res["meta"]["fluid_stage"]
            # Pre-registered fluid gates (G2, G7, P2), origins in the
            # pre-registration: flux imbalance at the convergence floor;
            # profile L2 an order above the measured level; recirculation
            # excluded with a wide margin.
            assert abs(fluid_stage["flux_imbalance_near"]) < 5e-3
            assert abs(fluid_stage["flux_imbalance_far"]) < 5e-3
            assert fluid_stage["mother_L2_vs_parabola"] < 2e-2
            assert fluid_stage["daughter_L2_vs_parabola"] < 2e-2
            u_min = float(fluid_stage["gates"]["u_min_bend_centreline"])
            assert u_min > 0.5 * 2.0 * float(
                case["physical"]["mean_velocity"]), (
                f"P2: bend centreline velocity fell to {u_min} — "
                f"recirculation or separation; a finding, report it")
        m = res["mass_over_initial"]
        assert m[0] == pytest.approx(1.0)
        # G1 bookkeeping: parking only, bounded, never growth. Floors by
        # wall: bounce-back parks -2.7% (straight record envelope); the
        # NT wall's cut-cell transit parks ~12% at this resolution
        # (measured, settling not draining - the drain signature was the
        # TRT+bounce-back failure).
        floor = 0.94 if wall == "bb" else 0.85
        assert m[-1] > floor, (
            f"leg {tag}: bulk mass fell to {m[-1]:.4f} (floor {floor})")
        assert m[-1] < 1.005, (
            f"leg {tag}: bulk mass GREW to {m[-1]:.4f} — a leak inward")
        legs[tag] = {
            "angle_deg": angle,
            "dynamics": dyn,
            "wall": wall,
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

    # THE IN-RUN CONTROL GATE (N30 window 1 vs N0 window 1): the first
    # window lies wholly upstream of the bend, so on the SAME instrument
    # the bend must not move it. Origins: output-sampling granularity on
    # the peak; the tail envelope.
    w1_bend = legs["N30"]["receivers"][0]
    w1_ctrl = legs["N0"]["receivers"][0]
    dpeak_w1 = w1_bend["peak_measured"]/w1_ctrl["peak_measured"] - 1.0
    dtail_w1 = (w1_bend["tail_ratio_3_to_6_t2"]
                - w1_ctrl["tail_ratio_3_to_6_t2"])
    assert abs(dpeak_w1) < 0.01, (
        f"in-run control: upstream window peak moved {dpeak_w1:+.2%} "
        f"under the bend — instrument, diagnose")
    assert abs(dtail_w1) < 0.05, (
        f"in-run control: upstream window tail moved {dtail_w1:+.3f}")

    # THE SOLVED-FLOW COST ON THE BEND (S30 vs N30): reported — the
    # analogue of the straight coupled model's 0.1-0.3% peak cost.
    solved_vs_prescribed = [{
        "dbar_um": sv["dbar_um"],
        "peak_relative": sv["peak_measured"]/pr["peak_measured"] - 1.0,
        "peak_lag_minus": sv["peak_lag_relative"] - pr["peak_lag_relative"],
        "tail_ratio_minus": (sv["tail_ratio_3_to_6_t2"]
                             - pr["tail_ratio_3_to_6_t2"]),
    } for sv, pr in zip(legs["S30"]["receivers"], legs["N30"]["receivers"])]

    # THE BEND MEASUREMENT (N30 vs N0, windows 2 and 3): reported.
    bend_vs_control = [{
        "dbar_um": b["dbar_um"],
        "peak_relative_to_control":
            b["peak_measured"]/c["peak_measured"] - 1.0,
        "peak_lag_minus_control":
            b["peak_lag_relative"] - c["peak_lag_relative"],
        "tail_ratio_minus_control":
            b["tail_ratio_3_to_6_t2"] - c["tail_ratio_3_to_6_t2"],
    } for b, c in zip(legs["N30"]["receivers"], legs["N0"]["receivers"])]

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
                "plain Bouzidi as scalar wall, AXIS-aligned: stable but "
                "retains only 0.25 of the scalar - the interpolation does "
                "not return the full outgoing population and a passive "
                "scalar has no pressure to self-correct; NOT mass-"
                "conserving, refuted as the no-flux candidate",
                "plain Bouzidi, oblique: leaks to 0.24 then diverges",
            ],
            "plane_wave_max_lambda_at_pin": rho,
            "resolution_chosen": "TRT, magic Lambda = 1/4 (Ginzburg's "
                                 "bounce-back stability optimum); D rides "
                                 "the odd rate, checked above, so the "
                                 "physics is untouched",
            "bend_leg_status": "BLOCKED pending a MASS-CONSERVING "
                               "interpolated no-flux ADE wall. Stock "
                               "OpenLB 1.9 ships only Bouzidi-ADE-"
                               "Dirichlet (absorbing); plain Bouzidi "
                               "reflection is refuted by the ladder "
                               "(not mass-conserving for a scalar). The "
                               "wall scheme is the named next rung; no "
                               "bend number is quoted before it lands.",
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
        "nt_wall": {
            "scheme": "Noble-Torczynski partially-saturated cells "
                      "(pairwise-antisymmetric solid operator, exactly "
                      "conservative) + TRT bulk at magic Lambda = 1/4; "
                      "either alone fails (NT-BGK diverges obliquely, "
                      "TRT+bounce-back drains to zero)",
            "instrument_calibration_N0_vs_straight_record":
                "peaks read high (TRT finite-k sharpening) and tail "
                "ratios read low (near-wall tail content sits partly in "
                "cut cells under bulk-only material accounting); these "
                "deltas are the readout shift of the new instrument, "
                "which is why the bend is quoted against N0, never "
                "against the record",
            "wall_layer_budget": "reference-lattice study: amp ~ "
                                 "5 u_lat = 0.2 over 2-3 cells at this "
                                 "resolution, shrinking ~ res^0.7 "
                                 "(results/oblique_wall_scheme_study"
                                 ".json)",
        },
        "bend_N30_vs_N0": bend_vs_control,
        "solved_flow": {
            "fluid_stage": fluid_stage,
            "S30_vs_N30": solved_vs_prescribed,
        },
        "mass_final_over_initial": {
            k: v["mass_final_over_initial"] for k, v in legs.items()},
        "meta": {k: v["meta"] for k, v in legs.items()},
        "git_sha": git_sha(REPO),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")
