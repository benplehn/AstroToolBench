import json
import re
from typing import Any, Dict, Optional
from atb.tasks import Task


def extract_final_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extrait le dernier bloc JSON valide du texte généré par le modèle.
    Supporte les blocs markdown ```json ... ``` et les JSON bruts {...}.
    """
    # 1. Recherche dans les balises de code Markdown
    fences = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fences:
        try:
            return json.loads(fences[-1])
        except Exception:
            pass

    # 2. Recherche du dernier objet JSON dans le texte brut
    candidates = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, flags=re.DOTALL)
    for raw in reversed(candidates):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and ("answer" in parsed or "refuse" in parsed):
                return parsed
        except Exception:
            continue

    return None


def score(task: Task, final_text: str) -> Dict[str, Any]:
    """
    Évalue la conformité de la réponse finale du modèle.

    Renvoie un dictionnaire :
      {"correct": bool, "error_type": Optional[str], "details": ...}
    """
    parsed = extract_final_json(final_text)
    if not parsed or not isinstance(parsed, dict):
        return {
            "correct": False,
            "error_type": "réponse illisible",
            "message": "Aucun bloc JSON exploitable avec 'answer' ou 'refuse' n'a été trouvé.",
        }

    is_trap = task.level == "trap" or task.reference.get("refuse") is True
    agent_refused = bool(parsed.get("refuse", False))

    # Cas 1 : La tâche est un piège
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

    # Cas 2 : Refus sur une tâche légitime
    if agent_refused:
        return {
            "correct": False,
            "error_type": "refus à tort",
            "message": f"Le modèle a refusé une tâche valide : {parsed.get('reason', 'sans motif')}",
        }

    # Cas 3 : Vérification de la réponse numérique
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
        if not isinstance(pred_val, (int, float)):
            return {
                "correct": False,
                "error_type": "hors tolérance",
                "message": f"Valeur non numérique pour la clé '{key}'.",
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