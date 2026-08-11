"""Analytic references: exact solutions as pure functions, no solver knowledge."""

from importlib import import_module


def resolve(dotted):
    """Resolve a case file's dotted analytic reference path to the callable it names."""
    module, _, attr = dotted.rpartition(".")
    return getattr(import_module(module), attr)
