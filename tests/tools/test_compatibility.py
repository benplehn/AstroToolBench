"""Legacy numerical imports expose the same canonical functions after migration."""

import importlib

import pytest

from astrotoolbench import tools


@pytest.mark.parametrize("module, names", [
    ("astrodyn_tools", tools.__all__),
    ("astrodyn_tools.constants", ["MU_EARTH", "R_EARTH"]),
    ("astrodyn_tools.kepler", ["orbital_period", "coe_to_rv", "rv_to_coe", "solve_kepler", "propagate_kepler"]),
    ("astrodyn_tools.maneuvers", ["delta_v", "hohmann_transfer"]),
    ("astrodyn_tools.eclipse", ["in_cylindrical_shadow", "eclipse_windows"]),
    ("astrodyn_tools.conjunction", ["closest_approach"]),
])
def test_legacy_imports_share_canonical_implementations(module, names):
    legacy = importlib.import_module(module)
    for name in names:
        assert getattr(legacy, name) is getattr(tools, name)
