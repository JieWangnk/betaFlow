"""betaflow — a solver-agnostic validation harness for haemodynamic CFD.

Layering (dependencies point downward only):

    tests/            pytest, one test per case
    betaflow/metrics  error norms — consume plain dicts of arrays
    betaflow/analytic references — pure functions, no solver knowledge
    betaflow/cases    YAML case definitions
    betaflow/runners  solver adapters — the ONLY layer that knows a solver exists

The runner contract is a plain dict:

    run_case(case, runner="openfoam") -> {"y": ndarray, "u": ndarray, "u_ref": float}

plus an optional "meta" sub-dict of provenance (solver version, cell counts).
Adding a second solver means adding one module in betaflow/runners/ and
changing nothing else.
"""

from betaflow.runners import run_case

__all__ = ["run_case"]
