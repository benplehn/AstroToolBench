"""Reviewed, deterministic family partitions and conservative equivalence groups.

Primitive tools are deliberately reusable in held-out compositions. Complete
successful computation recipes, numerical variants and exact problem duplicates
must not move across partitions. This is a local data policy, not a claim about
an external model's pretraining corpus.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

from .schema import BenchmarkTask


@dataclass(frozen=True)
class SplitGroup:
    split: str
    families: tuple[str, ...]
    rationale: str


GROUPS = {
    "kepler_propagation": SplitGroup("train", ("kepler_circular", "kepler_elliptic"),
        "Circular and elliptic states use the same complete Kepler propagation operation."),
    "impulsive_maneuvers": SplitGroup("train", ("velocity_impulse", "circularization"),
        "Vector impulses and local circularization share the velocity-change computation."),
    "circular_transfers": SplitGroup("train", ("hohmann_transfer", "transfer_comparison"),
        "Raising, lowering, equal radii and comparisons reuse the Hohmann solution."),
    "cylindrical_shadow": SplitGroup("train", ("eclipse_detection", "eclipse_duration"),
        "Instantaneous geometry and sampled intervals share the same shadow model."),
    "unit_contract": SplitGroup("train", ("invalid_units",),
        "Mixed position/velocity units are the same dimensional-consistency failure."),
    "state_separation": SplitGroup("validation", ("kepler_comparison", "separation_at_epoch"),
        "Both labels compute two propagated positions and their simultaneous separation."),
    "input_contract": SplitGroup("validation", ("missing_information", "invalid_search", "invalid_parameters"),
        "Missing information, invalid observation intervals and nonpositive gravity are development diagnostics."),
    "closest_approach": SplitGroup("test", ("closest_approach",),
        "A minimum over time is held out from instantaneous-separation questions."),
    "transfer_composition": SplitGroup("test", ("transfer_then_propagation", "transfer_then_eclipse", "transfer_arrival_eclipse"),
        "Dependent transfer/propagation/shadow workflows are held out as complete compositions."),
    "orbital_geometry": SplitGroup("test", ("invalid_geometry", "unsupported_orbit"),
        "Body intersection, undefined shadow geometry and unbound orbital states are held-out physical diagnoses."),
    "model_limits": SplitGroup("test", ("unsupported_physics", "precision_contract"),
        "Unavailable perturbation physics and unattainable event guarantees are held-out model-limit diagnoses."),
}

# A familiar label cannot disguise a different computation and bypass the policy.
# Failure families list relevant recipes and codes; unlike numerical successes,
# the same tool may legitimately have different, held-out failure mechanisms.
FAMILY_CONTRACTS = {
    "kepler_circular": ({"propagation"}, {"success"}),
    "kepler_elliptic": ({"propagation"}, {"success"}),
    "kepler_comparison": ({"separation"}, {"success"}),
    "velocity_impulse": ({"delta_v"}, {"success"}),
    "circularization": ({"circularization"}, {"success"}),
    "hohmann_transfer": ({"hohmann"}, {"success"}),
    "transfer_comparison": ({"compare_transfers"}, {"success"}),
    "eclipse_detection": ({"shadow"}, {"success"}),
    "eclipse_duration": ({"eclipse"}, {"success"}),
    "separation_at_epoch": ({"separation"}, {"success"}),
    "closest_approach": ({"proximity"}, {"success"}),
    "transfer_then_propagation": ({"transfer_propagation"}, {"success"}),
    "transfer_then_eclipse": ({"transfer_eclipse"}, {"success"}),
    "transfer_arrival_eclipse": ({"transfer_arrival_shadow"}, {"success"}),
    "invalid_units": ({"delta_v", "propagation"}, {"inconsistent_units"}),
    "unsupported_orbit": ({"propagation"}, {"invalid_input"}),
    "invalid_geometry": ({"eclipse", "propagation", "hohmann"}, {"invalid_input", "physical_impossibility"}),
    "missing_information": ({"eclipse"}, {"missing_input"}),
    "unsupported_physics": ({"propagation"}, {"unsupported_model"}),
    "invalid_search": ({"eclipse", "proximity"}, {"invalid_input"}),
    "precision_contract": ({"eclipse"}, {"unachievable_precision"}),
    "invalid_parameters": ({"hohmann"}, {"invalid_input"}),
}


def group_for_family(family: str) -> str:
    matches = [name for name, group in GROUPS.items() if family in group.families]
    if len(matches) != 1 or family not in FAMILY_CONTRACTS:
        raise ValueError(f"Family {family!r} needs exactly one reviewed split group and computation contract.")
    return matches[0]


def split_for_family(family: str) -> str:
    return GROUPS[group_for_family(family)].split


def _outcome(task):
    contract = task.expected if isinstance(task, BenchmarkTask) else task.answer_contract
    return "success" if contract.kind == "success" else contract.code


def problem_fingerprint(task) -> str:
    """Detect duplicates independently of wording, ID, output subset and scan settings.

    Units and physical context are retained; numerical values are canonicalized
    at 12 significant figures to also catch harmless float serialization changes.
    Group-level checks, rather than this hash, prevent radius-only variants from
    masquerading as new held-out families.
    """
    def canonical(value):
        if isinstance(value, list):
            return [canonical(component) for component in value]
        if isinstance(value, (int, float)):
            return 0.0 if value == 0 else float(format(value, ".12g"))
        return value

    physical_inputs = {name: {"value": canonical(quantity.value), "unit": quantity.unit}
                       for name, quantity in task.inputs.items() if name not in {"step", "tol"}}
    context = task.context.model_dump()
    # A requested guarantee is part of the question; the common epoch/frame is too.
    payload = {"recipe": task.recipe, "context": context, "inputs": physical_inputs}
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def audit_splits(tasks: Iterable) -> dict:
    """Fail closed on unknown families, split drift, disguised recipes or duplicates."""
    tasks = list(tasks)
    ids, fingerprints, successful_recipes = set(), {}, {}
    for task in tasks:
        if task.id in ids:
            raise ValueError(f"Duplicate task ID: {task.id}")
        ids.add(task.id)
        group_name = group_for_family(task.family)
        group = GROUPS[group_name]
        if task.split != group.split:
            raise ValueError(f"{task.id}: family {task.family} belongs to {group_name}/{group.split}, not {task.split}.")
        recipes, outcomes = FAMILY_CONTRACTS[task.family]
        outcome = _outcome(task)
        if task.recipe not in recipes or outcome not in outcomes:
            raise ValueError(f"{task.id}: recipe/outcome does not match reviewed family {task.family}.")
        if outcome == "success":
            previous = successful_recipes.setdefault(task.recipe, task.split)
            if previous != task.split:
                raise ValueError(f"Successful recipe {task.recipe} appears in multiple splits.")
        fingerprint = problem_fingerprint(task)
        previous = fingerprints.setdefault(fingerprint, (task.split, task.id))
        if previous[0] != task.split:
            raise ValueError(f"{task.id}: numerical duplicate of {previous[1]} across splits.")
    return {
        "policy_version": "1",
        "strategy": "reviewed_family_and_complete_computation_groups",
        "groups": {name: {"split": group.split, "families": list(group.families), "rationale": group.rationale}
                   for name, group in GROUPS.items()},
        "splits": {split: sorted(task.id for task in tasks if task.split == split)
                   for split in ("train", "validation", "test")},
        "cross_split_family_overlaps": 0,
        "cross_split_successful_recipe_overlaps": 0,
        "cross_split_problem_duplicates": 0,
        "scope": "Primitive tool reuse inside held-out compositions is intentional. Legacy development data is excluded from training. External pretraining contamination is not assessed.",
    }
