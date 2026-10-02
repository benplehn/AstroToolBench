import json
from typing import Any, Dict
import numpy as np

from astrodyn_tools.constants import MU_EARTH, R_EARTH
from astrodyn_tools.kepler import orbital_period, propagate_kepler
from astrodyn_tools.maneuvers import hohmann_transfer
from astrodyn_tools.eclipse import eclipse_windows


def _json_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serializable(v) for v in obj]
    return obj


def execute_tool(api_version: str, name: str, arguments_json: str) -> str:
    """
    Exécute un outil en version 'B' ou 'C'.
    Retourne une chaîne JSON valide contenant le résultat ou l'erreur.
    """
    api = api_version.upper()
    if api not in ("B", "C"):
        raise ValueError(f"Version d'API inconnue: {api_version}. Utiliser 'B' ou 'C'.")

    # 1. Parsing JSON des arguments
    try:
        args: Dict[str, Any] = json.loads(arguments_json)
    except Exception as e:
        if api == "B":
            return json.dumps({"error": f"JSONDecodeError: {str(e)}"})
        return json.dumps({"error": f"Malformed arguments JSON: {str(e)}. Please provide valid JSON."})

    try:
        if name == "orbital_period":
            if api == "B":
                a = float(args["a"])
                mu = float(args.get("mu", MU_EARTH))
                res = orbital_period(a, mu=mu)
                return json.dumps({"period": float(res)})
            else:
                alt = float(args["altitude_km"])
                if alt < 0:
                    return json.dumps({
                        "error": f"Invalid altitude: {alt} km. Altitude must be >= 0 km; did you pass a negative value or center radius?"
                    })
                r = R_EARTH + alt
                res = orbital_period(r, mu=MU_EARTH)
                return json.dumps({"period_s": float(res)})

        elif name == "hohmann_transfer":
            if api == "B":
                r1 = float(args["r1"])
                r2 = float(args["r2"])
                mu = float(args.get("mu", MU_EARTH))
                res = hohmann_transfer(r1, r2, mu=mu)
                return json.dumps(_json_serializable(res))
            else:
                h1 = float(args["initial_altitude_km"])
                h2 = float(args["final_altitude_km"])
                if h1 < 0 or h2 < 0:
                    return json.dumps({
                        "error": f"Invalid altitudes (h1={h1}, h2={h2}). Altitudes must be >= 0 km above Earth surface."
                    })
                r1 = R_EARTH + h1
                r2 = R_EARTH + h2
                res = hohmann_transfer(r1, r2, mu=MU_EARTH)
                return json.dumps({
                    "delta_v1_km_s": float(res["dv1"]),
                    "delta_v2_km_s": float(res["dv2"]),
                    "delta_v_total_km_s": float(res["dv_total"]),
                    "transfer_time_s": float(res["tof"])
                })

        elif name == "propagate_orbit":
            if api == "B":
                r0 = np.array(args["r0"], dtype=float)
                v0 = np.array(args["v0"], dtype=float)
                dt = float(args["dt"])
                mu = float(args.get("mu", MU_EARTH))
                rf, vf = propagate_kepler(r0, v0, dt, mu=mu)
                return json.dumps({"rf": rf.tolist(), "vf": vf.tolist()})
            else:
                r0 = np.array(args["r0_km"], dtype=float)
                v0 = np.array(args["v0_km_s"], dtype=float)
                dt = float(args["dt_s"])
                if np.linalg.norm(r0) < R_EARTH:
                    return json.dumps({
                        "error": f"Initial position radius ({np.linalg.norm(r0):.1f} km) is below Earth surface ({R_EARTH} km)."
                    })
                rf, vf = propagate_kepler(r0, v0, dt, mu=MU_EARTH)
                return json.dumps({
                    "r_final_km": rf.tolist(),
                    "v_final_km_s": vf.tolist()
                })

        elif name == "eclipse_windows":
            if api == "B":
                r0 = np.array(args["r0"], dtype=float)
                v0 = np.array(args["v0"], dtype=float)
                sun_dir = np.array(args["sun_dir"], dtype=float)
                duration = float(args["duration"])
                windows = eclipse_windows(r0, v0, sun_dir, duration)
                return json.dumps({"windows": _json_serializable(windows)})
            else:
                r0 = np.array(args["r0_km"], dtype=float)
                v0 = np.array(args["v0_km_s"], dtype=float)
                sun_dir = np.array(args["sun_dir"], dtype=float)
                duration = float(args["duration_s"])
                if duration <= 0:
                    return json.dumps({"error": f"duration_s must be strictly positive, got {duration}."})
                windows = eclipse_windows(r0, v0, sun_dir, duration)
                total_duration = sum(w[1] - w[0] for w in windows) if windows else 0.0
                return json.dumps({
                    "eclipse_windows_s": _json_serializable(windows),
                    "total_eclipse_duration_s": float(total_duration)
                })

        else:
            if api == "B":
                return json.dumps({"error": f"Unknown tool: '{name}'"})
            return json.dumps({
                "error": f"Tool '{name}' is not recognized. Available tools: orbital_period, hohmann_transfer, propagate_orbit, eclipse_windows."
            })

    except Exception as e:
        if api == "B":
            # Exception brute en B
            return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})
        # Message contextualisé en C
        return json.dumps({
            "error": f"Execution failed for tool '{name}' with error: {type(e).__name__}: {str(e)}."
        })