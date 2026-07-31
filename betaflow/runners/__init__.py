"""Solver adapters. The ONLY layer of betaflow allowed to know a solver exists.

Contract
--------
Each runner is one module in this package exposing

    run(case: dict, **params) -> dict

where `case` is the parsed YAML case definition and the returned dict carries
whatever the case's metrics consume, plus a `meta` sub-dict of provenance.

CORRECTED after the first non-OpenFOAM runner. This docstring previously
specified the return as {"y", "u", "u_ref", "meta"} — a FLUID-SPECIFIC
contract that quietly assumed a wall-normal velocity profile. The dispatch
below never enforced it, so `runners/langevin.py` (free Brownian motion,
returning {"t", "msd", "msd_components", "D_expected", ...}) plugged in with
no change anywhere above this layer. The layering claim held; only its
statement was too narrow.

What IS required of every runner: return a `meta` dict, and return arrays the
case's declared metrics can consume. Fluid cases conventionally return
y/u/u_ref, and that convention is worth keeping — but it is a convention of
those cases, not of the harness.
"""

from importlib import import_module


def run_case(case, runner="openfoam", **params):
    """Dispatch `case` to the named runner module and return its result dict."""
    try:
        module = import_module(f"betaflow.runners.{runner}")
    except ModuleNotFoundError as err:
        raise ValueError(
            f"unknown runner '{runner}': expected a module betaflow/runners/{runner}.py"
        ) from err
    return module.run(case, **params)
