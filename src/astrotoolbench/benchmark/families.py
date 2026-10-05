"""Authored problem families, not a Cartesian product of prompt paraphrases.

Inputs describe different geometry, limits and failure modes. Expected values
are computed by scientific recipes, then checked by independent certificates.
"""

from dataclasses import dataclass, field
import json
import math

from astrotoolbench.tools import MU_EARTH, R_EARTH

from .catalog import RECIPES
from .generation import generate_records
from .schema import BenchmarkTask, TaskSpecification
from .splits import split_for_family


@dataclass(frozen=True)
class Problem:
    id: str
    category: str
    family: str
    difficulty: str
    recipe: str
    question: str
    parameters: dict
    outputs: tuple[str, ...] = ()
    error: str | None = None
    reason: str = ""
    method: str = "analytic"
    rationale: str = "Solution analytique indépendante, détaillée dans docs/ground-truth.md."
    unit_overrides: dict[str, str] = field(default_factory=dict)
    dynamics: str = "two_body"
    requested_time_accuracy_s: float | None = None
    notes: str = "Référence calculée sans arrondi; toutes les unités sont explicites."


def _circle(radius=7000.0, phase=0.0, inclination=0.0):
    """Construct a circle directly in Cartesian coordinates, without coe_to_rv."""
    speed = math.sqrt(MU_EARTH / radius)
    c, s = math.cos(phase), math.sin(phase)
    ci, si = math.cos(inclination), math.sin(inclination)
    return {"r0": [radius*c, radius*s*ci, radius*s*si],
            "v0": [-speed*s, speed*c*ci, speed*c*si], "mu": MU_EARTH}


def _ellipse(a=10000.0, e=0.2, apogee=False):
    radius = a * (1+e if apogee else 1-e)
    speed = math.sqrt(MU_EARTH * (2/radius - 1/a))
    sign = -1 if apogee else 1
    return {"r0": [sign*radius, 0.0, 0.0], "v0": [0.0, sign*speed, 0.0], "mu": MU_EARTH}


def _encounter(meeting_time=500.0, inclination=math.pi/2):
    rate = math.sqrt(MU_EARTH / 7000**3)
    first = _circle(phase=-rate*meeting_time)
    second = _circle(phase=-rate*meeting_time, inclination=inclination)
    return {"r1": first["r0"], "v1": first["v0"], "r2": second["r0"], "v2": second["v0"], "mu": MU_EARTH}


def problem_definitions() -> list[Problem]:
    """Fifty reviewed problems with stable IDs and structural family labels."""
    period = 2*math.pi*math.sqrt(7000**3/MU_EARTH)
    search = {"step": 8.0, "tol": 1e-6}
    shadow = {"sun_dir": [1.0, 0.0, 0.0], "r_body": R_EARTH}
    eclipse = _circle() | shadow | search
    transfer = {"r1": 7000.0, "r2": 9000.0, "mu": MU_EARTH}
    crossing = _encounter() | search | {"window": 1000.0}
    circularization = "La vitesse cible est tangentielle dans le sens du mouvement; conserver position et plan."
    return [
        # A: propagation, including signed time, elliptic motion and a comparison.
        Problem("prop-circular-quarter", "propagation", "kepler_circular", "simple", "propagation",
                "Propager pendant un quart de période et donner position et vitesse finales.",
                _circle() | {"dt": period/4}, ("position", "velocity")),
        Problem("prop-backward", "propagation", "kepler_circular", "simple", "propagation",
                "Propager l'état vers le passé. Une durée signée négative est autorisée ici.",
                _circle() | {"dt": -period/4}, ("position", "velocity")),
        Problem("prop-periapsis-apogee", "propagation", "kepler_elliptic", "simple", "propagation",
                "Propager depuis le périapside pendant une demi-période de l'ellipse.",
                _ellipse() | {"dt": math.pi*math.sqrt(10000**3/MU_EARTH)}, ("position", "velocity"), method="cartesian_rk4",
                rationale="Intégration cartésienne RK4 indépendante et contrôle du passage à l'apoapside."),
        Problem("prop-elliptic-arbitrary", "propagation", "kepler_elliptic", "simple", "propagation",
                "Propager cette ellipse pendant la durée donnée, sans arrondir les résultats intermédiaires.",
                _ellipse(11000, 0.25, apogee=True) | {"dt": 1234.5}, ("position", "velocity"), method="cartesian_rk4",
                rationale="Intégration directe des équations cartésiennes, indépendante du solveur de Kepler."),
        Problem("prop-inclined-full-period", "propagation", "kepler_circular", "simple", "propagation",
                "Propager l'orbite circulaire inclinée pendant une période complète.",
                _circle(inclination=0.8) | {"dt": period}, ("position", "velocity")),
        Problem("prop-zero-duration", "propagation", "kepler_circular", "simple", "propagation",
                "Propager pendant zéro seconde et donner l'état final.",
                _circle(phase=0.3) | {"dt": 0.0}, ("position", "velocity")),
        Problem("prop-compare-two-states", "propagation", "kepler_comparison", "multistep", "separation",
                "Propager séparément les deux états jusqu'au même instant, puis calculer leur séparation.",
                _encounter(meeting_time=600.0) | {"dt": 300.0}, ("distance",)),
        Problem("prop-unbound-state", "propagation", "unsupported_orbit", "trap", "propagation",
                "Peut-on propager cet état avec le propagateur elliptique fourni? Justifier une éventuelle erreur.",
                _circle() | {"v0": [0.0, 12.0, 0.0], "dt": 100.0}, error="invalid_input",
                reason="L'énergie spécifique est positive; le propagateur est limité aux ellipses."),
        # B: vector impulses, raising/lowering, comparisons and circularization.
        Problem("dv-vector-turn", "maneuvers", "velocity_impulse", "simple", "delta_v",
                "Calculer la norme de l'impulsion, en tenant compte du changement de direction.",
                {"v_initial": [3.0, 0.0, 0.0], "v_final": [0.0, 4.0, 0.0]}, ("delta_v",)),
        Problem("dv-reversal", "maneuvers", "velocity_impulse", "simple", "delta_v",
                "Calculer le ΔV nécessaire pour inverser la vitesse instantanément.",
                {"v_initial": [0.0, 7.5, 0.0], "v_final": [0.0, -7.5, 0.0]}, ("delta_v",)),
        Problem("hohmann-raising", "maneuvers", "hohmann_transfer", "simple", "hohmann",
                "Calculer les deux impulsions, le ΔV total et le temps du transfert entre orbites circulaires coplanaires.",
                transfer, ("dv1", "dv2", "dv_total", "tof")),
        Problem("hohmann-lowering", "maneuvers", "hohmann_transfer", "simple", "hohmann",
                "Calculer le transfert descendant. Retourner des magnitudes positives pour les impulsions.",
                transfer | {"r1": 42164.0, "r2": 7000.0}, ("dv1", "dv2", "dv_total", "tof")),
        Problem("hohmann-compare-targets", "maneuvers", "transfer_comparison", "multistep", "compare_transfers",
                "Calculer deux transferts depuis le même rayon puis la différence ΔV(B) - ΔV(A).",
                {"r_start": 7000.0, "r_target_a": 9000.0, "r_target_b": 14000.0, "mu": MU_EARTH},
                ("dv_a", "dv_b", "difference")),
        Problem("dv-circularize-periapsis", "maneuvers", "circularization", "multistep", "circularization",
                "Calculer la vitesse circulaire locale puis l'impulsion de circularisation. " + circularization,
                _ellipse(), ("delta_v",)),
        Problem("dv-circularize-apogee", "maneuvers", "circularization", "multistep", "circularization",
                "Circulariser à l'apoapside à partir de l'état fourni. " + circularization,
                _ellipse(apogee=True), ("delta_v",)),
        Problem("dv-mixed-units", "maneuvers", "invalid_units", "trap", "delta_v",
                "Utiliser directement ces vecteurs sans convertir leurs unités. Signaler si ce calcul est incohérent.",
                {"v_initial": [0.0, 7500.0, 0.0], "v_final": [0.0, 8.0, 0.0]}, error="inconsistent_units",
                reason="Le premier vecteur est en m/s, le contrat exige km/s; conversion explicitement interdite.",
                unit_overrides={"v_initial": "m/s"}),
        # C: illuminated/dark positions, full/partial windows and impossible geometry.
        Problem("shadow-day-side", "eclipse", "eclipse_detection", "simple", "shadow",
                "Déterminer si cette position est dans l'ombre cylindrique.",
                shadow | {"r_sat": [7000.0, 0.0, 0.0]}, ("in_shadow",)),
        Problem("shadow-night-side", "eclipse", "eclipse_detection", "simple", "shadow",
                "Déterminer si cette position antisolaire est dans l'ombre.",
                shadow | {"r_sat": [-7000.0, 0.0, 0.0]}, ("in_shadow",)),
        Problem("eclipse-one-period", "eclipse", "eclipse_duration", "simple", "eclipse",
                "Calculer les frontières des intervalles d'éclipse et leur durée totale sur une période.",
                eclipse | {"duration": period}, ("boundaries", "total_duration")),
        Problem("eclipse-two-periods", "eclipse", "eclipse_duration", "simple", "eclipse",
                "Calculer tous les intervalles d'éclipse sur deux périodes, puis leur durée cumulée.",
                eclipse | {"duration": 2*period}, ("boundaries", "total_duration")),
        Problem("eclipse-clipped-window", "eclipse", "eclipse_duration", "diagnostic", "eclipse",
                "L'observation débute en éclipse; ne compter que la durée dans la fenêtre et tronquer les intervalles.",
                _circle(phase=math.pi) | shadow | search | {"duration": 200.0}, ("boundaries", "total_duration")),
        Problem("eclipse-zero-window", "eclipse", "eclipse_duration", "diagnostic", "eclipse",
                "Une fenêtre nulle est autorisée. Calculer sa durée totale d'éclipse.",
                eclipse | {"duration": 0.0}, ("total_duration",)),
        Problem("eclipse-dawn-dusk", "eclipse", "eclipse_duration", "diagnostic", "eclipse",
                "Le Soleil est normal au plan de cette orbite. Évaluer la durée d'éclipse.",
                eclipse | {"sun_dir": [0.0, 0.0, 1.0], "duration": period}, ("total_duration",)),
        Problem("eclipse-zero-sun-direction", "eclipse", "invalid_geometry", "trap", "eclipse",
                "Évaluer si la direction solaire fournie permet de définir une ombre.",
                eclipse | {"sun_dir": [0.0, 0.0, 0.0], "duration": 1000.0}, error="invalid_input",
                reason="Un vecteur solaire nul ne définit aucune direction."),
        # D: instantaneous distance, interior/boundary minima and no useful approach.
        Problem("proximity-instant", "proximity", "separation_at_epoch", "simple", "separation",
                "Calculer la séparation simultanée à l'instant spécifié.",
                _encounter() | {"dt": 250.0}, ("distance",)),
        Problem("proximity-interior-crossing", "proximity", "closest_approach", "simple", "proximity",
                "Chercher le rapprochement minimal simultané dans l'intervalle fermé.",
                crossing, ("tca", "miss_distance")),
        Problem("proximity-start-boundary", "proximity", "closest_approach", "simple", "proximity",
                "Inclure la borne initiale dans la recherche de distance minimale.",
                _encounter(0.0) | search | {"window": 800.0}, ("tca", "miss_distance")),
        Problem("proximity-end-boundary", "proximity", "closest_approach", "simple", "proximity",
                "Inclure la borne finale dans la recherche de distance minimale.",
                _encounter(800.0) | search | {"window": 800.0}, ("tca", "miss_distance")),
        Problem("proximity-constant-separation", "proximity", "closest_approach", "simple", "proximity",
                "Ces objets partagent une orbite circulaire avec déphasage fixe. Calculer la distance minimale; aucun instant unique n'est demandé.",
                {"r1": _circle()["r0"], "v1": _circle()["v0"], "r2": _circle(phase=0.6)["r0"],
                 "v2": _circle(phase=0.6)["v0"], "mu": MU_EARTH, "window": 300.0} | search, ("miss_distance",)),
        Problem("proximity-zero-window", "proximity", "closest_approach", "diagnostic", "proximity",
                "Une fenêtre nulle est autorisée: retourner l'instant initial et la séparation initiale.",
                crossing | {"window": 0.0}, ("tca", "miss_distance")),
        Problem("proximity-identical-trajectories", "proximity", "closest_approach", "diagnostic", "proximity",
                "Les trajectoires sont identiques. Retourner la distance minimale et le premier instant qui la réalise.",
                _encounter(100.0, inclination=0.0) | search | {"window": 100.0}, ("tca", "miss_distance")),
        Problem("proximity-negative-window", "proximity", "invalid_search", "trap", "proximity",
                "Peut-on chercher un minimum sur cette fenêtre d'observation?",
                crossing | {"window": -10.0}, error="invalid_input", reason="Une fenêtre de recherche doit être non négative."),
        # E: genuine dependencies; every requested intermediate result is scored.
        Problem("multi-transfer-propagate-circle", "multistep", "transfer_then_propagation", "multistep", "transfer_propagation",
                "Calculer le transfert, puis propager l'état de départ pendant son temps de vol, SANS appliquer ses impulsions.",
                transfer | _circle(), ("tof", "dv_total", "position"), method="cartesian_rk4"),
        Problem("multi-transfer-propagate-ellipse", "multistep", "transfer_then_propagation", "multistep", "transfer_propagation",
                "Calculer le transfert circulaire indiqué, puis utiliser son temps de vol pour propager l'état elliptique indépendant fourni, sans impulsion.",
                transfer | _ellipse(), ("tof", "dv_total", "position"), method="cartesian_rk4"),
        Problem("multi-transfer-propagate-lowering", "multistep", "transfer_then_propagation", "multistep", "transfer_propagation",
                "Calculer ce transfert descendant, puis propager l'état circulaire fourni pendant la même durée, sans appliquer de manœuvre.",
                transfer | {"r1": 9000.0, "r2": 7000.0} | _circle(9000), ("tof", "dv_total", "position"), method="cartesian_rk4"),
        Problem("multi-transfer-eclipse-day-start", "multistep", "transfer_then_eclipse", "multistep", "transfer_eclipse",
                "Calculer le temps du transfert, puis la durée cumulée d'éclipse de l'orbite initiale non manœuvrée sur cette fenêtre.",
                transfer | eclipse, ("tof", "total_duration")),
        Problem("multi-transfer-eclipse-night-start", "multistep", "transfer_then_eclipse", "multistep", "transfer_eclipse",
                "Calculer le transfert puis utiliser son temps de vol comme fenêtre d'éclipse de l'état fourni, sans appliquer les impulsions.",
                transfer | _circle(phase=math.pi) | shadow | search, ("tof", "total_duration")),
        Problem("multi-transfer-eclipse-lowering", "multistep", "transfer_then_eclipse", "multistep", "transfer_eclipse",
                "Calculer ce transfert descendant et la durée d'éclipse de l'orbite de départ non manœuvrée pendant son temps de vol.",
                transfer | {"r1": 9000.0, "r2": 7000.0} | _circle(9000) | shadow | search, ("tof", "total_duration")),
        Problem("multi-arrival-in-shadow", "multistep", "transfer_arrival_eclipse", "multistep", "transfer_arrival_shadow",
                "Départ en +x, vitesse circulaire vers +y. Appliquer la première impulsion du transfert, propager jusqu'à l'arrivée, puis déterminer l'éclipse.",
                transfer | shadow, ("tof", "arrival_position", "in_shadow")),
        Problem("multi-arrival-illuminated", "multistep", "transfer_arrival_eclipse", "multistep", "transfer_arrival_shadow",
                "Départ en +x, vitesse circulaire vers +y. Effectuer le transfert descendant et tester l'ombre à l'arrivée avec cette direction solaire.",
                transfer | {"r1": 9000.0, "r2": 7000.0} | shadow | {"sun_dir": [-1.0, 0.0, 0.0]},
                ("tof", "arrival_position", "in_shadow")),
        # F: independently checkable omissions, model limits and invalid physics.
        Problem("diagnostic-missing-state", "diagnostic", "missing_information", "diagnostic", "eclipse",
                "Peut-on déterminer l'éclipse si la vitesse initiale n'est pas fournie? Ne pas l'inventer.",
                eclipse | {"v0": None, "duration": 1000.0}, error="missing_input", reason="La vitesse initiale est nécessaire."),
        Problem("diagnostic-j2-request", "diagnostic", "unsupported_physics", "diagnostic", "propagation",
                "Propager en incluant J2 avec les outils du contrat actuel. Indiquer si le modèle demandé est disponible.",
                _circle() | {"dt": 600.0}, error="unsupported_model", reason="Les outils ne modélisent que le problème à deux corps.", dynamics="J2"),
        Problem("trap-meters-as-kilometers", "diagnostic", "invalid_units", "trap", "propagation",
                "Passer directement cette position au propagateur sans conversion d'unités; évaluer la cohérence.",
                _circle() | {"r0": [7000000.0, 0.0, 0.0], "dt": 600.0}, error="inconsistent_units",
                reason="La position est en mètres; le contrat exige des kilomètres et interdit ici une conversion implicite.", unit_overrides={"r0": "m"}),
        Problem("trap-body-intersecting-ellipse", "diagnostic", "invalid_geometry", "trap", "propagation",
                "Cette orbite doit rester à l'extérieur de la Terre sur toute sa trajectoire. Vérifier sa validité avant propagation.",
                _ellipse(9000, 0.4, apogee=True) | {"dt": 600.0}, error="physical_impossibility",
                reason="Le périapside de l'ellipse est inférieur au rayon terrestre, bien que l'état initial soit extérieur."),
        Problem("trap-negative-observation-duration", "diagnostic", "invalid_search", "trap", "eclipse",
                "Évaluer l'éclipse sur la fenêtre fournie. Distinguer une fenêtre d'observation d'une durée de propagation signée.",
                eclipse | {"duration": -1.0}, error="invalid_input", reason="Une durée d'observation est non négative."),
        Problem("trap-unachievable-time-precision", "diagnostic", "precision_contract", "trap", "eclipse",
                "Garantir les instants d'éclipse à 1e-12 s sans changer la tolérance configurée. Évaluer cette exigence.",
                eclipse | {"duration": period}, error="unachievable_precision",
                reason="La largeur de bracket configurée à 1e-6 s ne garantit pas une précision de 1e-12 s.", requested_time_accuracy_s=1e-12),
        Problem("trap-underground-transfer", "diagnostic", "invalid_geometry", "trap", "hohmann",
                "Ce transfert décrit des satellites extérieurs à la Terre. Vérifier les rayons avant tout calcul.",
                transfer | {"r1": 2500.0}, error="physical_impossibility", reason="Le rayon initial est sous la surface terrestre."),
        Problem("trap-zero-gravity", "diagnostic", "invalid_parameters", "trap", "hohmann",
                "Calculer ce transfert si les paramètres satisfont le contrat scientifique.",
                transfer | {"mu": 0.0}, error="invalid_input", reason="Le paramètre gravitationnel doit être strictement positif."),
        # Milestone 5: two meaningful limits bring the corpus to fifty problems.
        Problem("hohmann-equal-radii", "maneuvers", "hohmann_transfer", "simple", "hohmann",
                "Les rayons sont identiques. Utiliser la convention Hohmann du contrat: impulsions nulles et temps égal à une demi-période, pas zéro seconde.",
                transfer | {"r2": 7000.0}, ("dv1", "dv2", "dv_total", "tof")),
        Problem("diagnostic-zero-impulse", "diagnostic", "velocity_impulse", "diagnostic", "delta_v",
                "Les deux vitesses sont identiques. Calculer le ΔV sans confondre une impulsion nulle avec une entrée invalide.",
                {"v_initial": [0.0, 7.5, 0.0], "v_final": [0.0, 7.5, 0.0]}, ("delta_v",)),
    ]


# Public scoring budgets and tighter independent verification budgets, by output.
_BUDGETS = {
    "position": (1e-3, 5e-5), "velocity": (1e-6, 5e-8), "arrival_position": (1e-3, 5e-5),
    "delta_v": (1e-6, 1e-10), "dv1": (1e-6, 1e-10), "dv2": (1e-6, 1e-10),
    "dv_total": (1e-6, 1e-10), "dv_a": (1e-6, 1e-10), "dv_b": (1e-6, 1e-10),
    "difference": (1e-6, 1e-10), "tof": (1e-3, 1e-8), "total_duration": (5e-3, 5e-6),
    "boundaries": (5e-3, 5e-6), "distance": (1e-3, 1e-7),
    "tca": (1e-3, 2e-6), "miss_distance": (1e-3, 2e-5),
}


def _specification(problem: Problem) -> TaskSpecification:
    recipe = RECIPES[problem.recipe]
    inputs = {name: {"value": value, "unit": problem.unit_overrides.get(name, recipe.inputs[name])}
              for name, value in problem.parameters.items()}
    expected_outputs, verification_tolerances = {}, {}
    for name in problem.outputs:
        if name == "in_shadow":
            expected_outputs[name] = {"kind": "boolean", "unit": "1"}
        else:
            scoring, verification = _BUDGETS[name]
            expected_outputs[name] = {"kind": "numeric", "unit": recipe.outputs[name],
                                      "absolute_tolerance": scoring, "relative_tolerance": None}
            verification_tolerances[name] = verification
    if problem.error is not None:
        expected = {"kind": "error", "code": problem.error, "reason": problem.reason}
    else:
        expected = {"kind": "success", "outputs": expected_outputs}
    parameters_text = "\n".join(f"- {name}: {json.dumps(quantity['value'])} {quantity['unit']}" for name, quantity in inputs.items())
    output_text = ", ".join(f"{name} [{recipe.outputs[name]}]" for name in problem.outputs)
    response = f"Retourner les sorties: {output_text}." if problem.outputs else "Si le calcul est impossible, retourner un code d'erreur et une justification; ne pas inventer de résultat."
    data = dict(
        schema_version="0.1", id=problem.id, domain="astrodynamics", category=problem.category,
        difficulty=problem.difficulty, split=split_for_family(problem.family), family=problem.family,
        prompt=f"{problem.question}\nRéférentiel ECI, époque commune t=0 s, Terre de rayon {R_EARTH} km. Modèle demandé: {problem.dynamics}. Soleil fixe, ombre cylindrique.\n{parameters_text}\n{response}",
        context=dict(frame="ECI", epoch_s=0.0, body="Earth", body_radius_km=R_EARTH,
                     dynamics=problem.dynamics, shadow="cylindrical_fixed_sun",
                     requested_time_accuracy_s=problem.requested_time_accuracy_s),
        inputs=inputs, recipe=problem.recipe, outputs=list(problem.outputs), required_tools=list(recipe.tools),
        answer_contract=expected, verification=dict(
            method="independent_input_check" if problem.error else problem.method,
            rationale=problem.reason if problem.error else problem.rationale,
            absolute_tolerances=verification_tolerances,
        ), notes=problem.notes,
    )
    return TaskSpecification.model_validate(data)


def build_specifications() -> list[TaskSpecification]:
    """Author task inputs, answer contracts and split metadata without computing answers."""
    return [_specification(problem) for problem in problem_definitions()]


def build_tasks() -> list[BenchmarkTask]:
    """Compatibility authoring helper: generate and independently verify all references."""
    return generate_records(build_specifications()).tasks
