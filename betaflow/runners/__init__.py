"""Solver adapters. The ONLY layer of betaflow allowed to know a solver exists.

Contract
--------
Each runner is one module in this package exposing

    run(case: dict, **params) -> {"y": ndarray, "u": ndarray, "u_ref": float,
                                  "meta": dict}

where `case` is the parsed YAML case definition, `y`/`u` are the sampled
wall-normal profile in SI units, `u_ref` is the reference velocity the oracle
non-dimensionalises by, and `meta` holds provenance (solver version, cell
counts, viscosity). Metrics and tests consume only that dict — adding a new
solver means adding one module here and changing nothing else.
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
