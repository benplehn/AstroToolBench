"""Parse final answers and compare them with benchmark references."""

import json
import math
from typing import Any
from atb.tasks import Task


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def extract_final_json(text: str) -> dict[str, Any] | None:
    """Find a final-answer JSON object in plain text or a Markdown code block."""
    decoder = json.JSONDecoder()
    final = None
    offset = 0
    while (start := text.find("{", offset)) != -1:
        try:
            parsed, end = decoder.raw_decode(text, start)
            if isinstance(parsed, dict) and ("answer" in parsed or "refuse" in parsed):
                final = parsed
            # Skip nested objects and braces inside strings in a parsed object.
            offset = end
        except json.JSONDecodeError:
            offset = start + 1
    return final


def score(task: Task, final_text: str) -> dict[str, Any]:
    """Return correctness, an error category and a message for a final answer."""
    parsed = extract_final_json(final_text)
    if not parsed or not isinstance(parsed, dict):
        return {
            "correct": False,
            "error_type": "réponse illisible",
            "message": "Aucun bloc JSON exploitable avec 'answer' ou 'refuse' n'a été trouvé.",
        }

    is_trap = task.level == "trap" or task.reference.get("refuse") is True
    agent_refused = parsed.get("refuse") is True

    if is_trap:
        if agent_refused:
            return {
                "correct": True,
                "error_type": None,
                "message": f"Refus justifié: {parsed.get('reason', '')}",
            }
        return {
            "correct": False,
            "error_type": "refus manquant",
            "message": "La tâche demandait un refus (piège/incohérence), mais le modèle a tenté de calculer.",
        }

    if agent_refused:
        return {
            "correct": False,
            "error_type": "refus à tort",
            "message": f"Le modèle a refusé une tâche valide : {parsed.get('reason', 'sans motif')}",
        }

    answer = parsed.get("answer")
    if not isinstance(answer, dict):
        return {
            "correct": False,
            "error_type": "clé manquante",
            "message": "Le bloc 'answer' est manquant ou n'est pas un dictionnaire.",
        }

    for key, target_val in task.reference.items():
        if key == "refuse":
            continue

        if key not in answer:
            return {
                "correct": False,
                "error_type": "clé manquante",
                "message": f"La clé attendue '{key}' est absente de la réponse.",
            }

        pred_val = answer[key]
        if not _is_finite_number(pred_val):
            return {
                "correct": False,
                "error_type": "hors tolérance",
                "message": f"Valeur numérique finie attendue pour la clé '{key}'.",
            }

        tol = task.tolerance.get(key, 1e-4)
        if abs(float(pred_val) - float(target_val)) > tol:
            return {
                "correct": False,
                "error_type": "hors tolérance",
                "message": f"Clé '{key}' hors tolérance: attendu={target_val}, reçu={pred_val} (tol={tol})",
            }

    return {
        "correct": True,
        "error_type": None,
        "message": "Réponse conforme à la vérité terrain.",
    }
