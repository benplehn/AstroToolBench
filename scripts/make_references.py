import json
from pathlib import Path
import numpy as np

from astrodyn_tools.constants import MU_EARTH, R_EARTH
from astrodyn_tools.kepler import orbital_period, propagate_kepler
from astrodyn_tools.maneuvers import hohmann_transfer
from astrodyn_tools.eclipse import eclipse_windows

SUN_X_NEG = [-1.0, 0.0, 0.0]

TASKS_SPECS = [
    # --- PÉRIODE ORBITALE (4 simples) ---
    {
        "id": "period-001",
        "family": "period",
        "level": "simple",
        "prompt": "Calculer la période orbitale (en secondes) d'un satellite en orbite circulaire à 400 km d'altitude autour de la Terre.",
        "expected_tools": ["orbital_period"],
        "params": {"altitude_km": 400.0},
        "target": "period_s",
        "tol": 1.0,
    },
    {
        "id": "period-002",
        "family": "period",
        "level": "simple",
        "prompt": "Calculer la période orbitale (en secondes) d'une orbite circulaire à 800 km d'altitude.",
        "expected_tools": ["orbital_period"],
        "params": {"altitude_km": 800.0},
        "target": "period_s",
        "tol": 1.0,
    },
    {
        "id": "period-003",
        "family": "period",
        "level": "simple",
        "prompt": "Calculer la période orbitale (en secondes) pour une orbite circulaire de rayon r = 7000 km.",
        "expected_tools": ["orbital_period"],
        "params": {"radius_km": 7000.0},
        "target": "period_s",
        "tol": 1.0,
    },
    {
        "id": "period-004",
        "family": "period",
        "level": "simple",
        "prompt": "Calculer la période orbitale d'un satellite géostationnaire (rayon r = 42164 km).",
        "expected_tools": ["orbital_period"],
        "params": {"radius_km": 42164.0},
        "target": "period_s",
        "tol": 5.0,
    },

    # --- TRANSFERT DE HOHMANN (4 simples) ---
    {
        "id": "hoh-001",
        "family": "hohmann",
        "level": "simple",
        "prompt": "Calculer le delta-v total pour un transfert de Hohmann entre un rayon initial de 7000 km et un rayon cible de 42164 km.",
        "expected_tools": ["hohmann_transfer"],
        "params": {"r1": 7000.0, "r2": 42164.0},
        "target": "delta_v_total_km_s",
        "tol": 0.01,
    },
    {
        "id": "hoh-002",
        "family": "hohmann",
        "level": "simple",
        "prompt": "Calculer la durée de transfert (tof en secondes) pour un transfert de Hohmann entre r1 = 6600 km et r2 = 7200 km.",
        "expected_tools": ["hohmann_transfer"],
        "params": {"r1": 6600.0, "r2": 7200.0},
        "target": "transfer_time_s",
        "tol": 2.0,
    },
    {
        "id": "hoh-003",
        "family": "hohmann",
        "level": "simple",
        "prompt": "Calculer le premier incrément de vitesse (dv1 en km/s) d'un transfert de Hohmann de r1 = 6700 km vers r2 = 35786 km.",
        "expected_tools": ["hohmann_transfer"],
        "params": {"r1": 6700.0, "r2": 35786.0},
        "target": "dv1_km_s",
        "tol": 0.01,
    },
    {
        "id": "hoh-004",
        "family": "hohmann",
        "level": "simple",
        "prompt": "Calculer le second incrément de vitesse (dv2 en km/s) d'un transfert de Hohmann entre r1 = 6800 km et r2 = 12000 km.",
        "expected_tools": ["hohmann_transfer"],
        "params": {"r1": 6800.0, "r2": 12000.0},
        "target": "dv2_km_s",
        "tol": 0.01,
    },

    # --- PROPAGATION (4 simples) ---
    {
        "id": "prop-001",
        "family": "propagation",
        "level": "simple",
        "prompt": "Propager l'état r0=[7000.0, 0.0, 0.0] km, v0=[0.0, 7.54605, 0.0] km/s pendant 1500 s. Donner la coordonnée x finale.",
        "expected_tools": ["propagate_orbit"],
        "params": {"r0": [7000.0, 0.0, 0.0], "v0": [0.0, 7.54605, 0.0], "dt": 1500.0},
        "target": "r_final_x_km",
        "tol": 2.0,
    },
    {
        "id": "prop-002",
        "family": "propagation",
        "level": "simple",
        "prompt": "Propager l'état r0=[7000.0, 0.0, 0.0] km, v0=[0.0, 7.54605, 0.0] km/s pendant 1500 s. Donner la coordonnée y finale.",
        "expected_tools": ["propagate_orbit"],
        "params": {"r0": [7000.0, 0.0, 0.0], "v0": [0.0, 7.54605, 0.0], "dt": 1500.0},
        "target": "r_final_y_km",
        "tol": 2.0,
    },
    {
        "id": "prop-003",
        "family": "propagation",
        "level": "simple",
        "prompt": "Propager l'état r0=[0.0, 8000.0, 0.0] km, v0=[-7.058, 0.0, 0.0] km/s pendant 3000 s. Donner la coordonnée x finale.",
        "expected_tools": ["propagate_orbit"],
        "params": {"r0": [0.0, 8000.0, 0.0], "v0": [-7.058, 0.0, 0.0], "dt": 3000.0},
        "target": "r_final_x_km",
        "tol": 2.0,
    },
    {
        "id": "prop-004",
        "family": "propagation",
        "level": "simple",
        "prompt": "Propager l'état r0=[0.0, 8000.0, 0.0] km, v0=[-7.058, 0.0, 0.0] km/s pendant 3000 s. Donner la coordonnée y finale.",
        "expected_tools": ["propagate_orbit"],
        "params": {"r0": [0.0, 8000.0, 0.0], "v0": [-7.058, 0.0, 0.0], "dt": 3000.0},
        "target": "r_final_y_km",
        "tol": 2.0,
    },

    # --- ÉCLIPSES (3 simples) ---
    {
        "id": "ecl-001",
        "family": "eclipse",
        "level": "simple",
        "prompt": "Calculer la durée totale d'éclipse sur une fenêtre de 6000 s pour r0=[7000.0, 0.0, 0.0] km, v0=[0.0, 7.546, 0.0] km/s avec le Soleil dans la direction [-1, 0, 0].",
        "expected_tools": ["eclipse_windows"],
        "params": {"r0": [7000.0, 0.0, 0.0], "v0": [0.0, 7.546, 0.0], "sun_dir": SUN_X_NEG, "duration": 6000.0},
        "target": "total_eclipse_duration_s",
        "tol": 10.0,
    },
    {
        "id": "ecl-002",
        "family": "eclipse",
        "level": "simple",
        "prompt": "Calculer la durée d'éclipse sur 7000 s pour une orbite circulaire à 8000 km de rayon (r0=[8000.0, 0.0, 0.0] km, v0=[0.0, 7.058, 0.0] km/s) avec soleil [-1, 0, 0].",
        "expected_tools": ["eclipse_windows"],
        "params": {"r0": [8000.0, 0.0, 0.0], "v0": [0.0, 7.058, 0.0], "sun_dir": SUN_X_NEG, "duration": 7000.0},
        "target": "total_eclipse_duration_s",
        "tol": 15.0,
    },
    {
        "id": "ecl-003",
        "family": "eclipse",
        "level": "simple",
        "prompt": "Calculer la durée totale d'éclipse sur 86400 s (24 h) d'une orbite géostationnaire r0=[42164.0, 0.0, 0.0], v0=[0.0, 3.0746, 0.0], sun_dir=[-1, 0, 0].",
        "expected_tools": ["eclipse_windows"],
        "params": {"r0": [42164.0, 0.0, 0.0], "v0": [0.0, 3.0746, 0.0], "sun_dir": SUN_X_NEG, "duration": 86400.0},
        "target": "total_eclipse_duration_s",
        "tol": 30.0,
    },

    # --- MULTI-ÉTAPES (4 tâches) ---
    {
        "id": "multi-001",
        "family": "multi_step",
        "level": "multi_step",
        "prompt": "Calculer le temps de vol d'un transfert de Hohmann entre 7000 km et 9000 km, puis propager l'orbite de départ pendant cette durée. Donner la position x finale.",
        "expected_tools": ["hohmann_transfer", "propagate_orbit"],
        "params": {"r1": 7000.0, "r2": 9000.0},
        "target": "r_final_x_km",
        "tol": 5.0,
    },
    {
        "id": "multi-002",
        "family": "multi_step",
        "level": "multi_step",
        "prompt": "Calculer la durée du transfert de Hohmann entre 6800 km et 7500 km, et calculer la période orbitale de l'orbite d'arrivée.",
        "expected_tools": ["hohmann_transfer", "orbital_period"],
        "params": {"r1": 6800.0, "r2": 7500.0},
        "target": "target_period_s",
        "tol": 2.0,
    },
    {
        "id": "multi-003",
        "family": "multi_step",
        "level": "multi_step",
        "prompt": "Calculer la période d'une orbite à 500 km d'altitude, puis calculer le delta-v total nécessaire pour monter à 1000 km d'altitude.",
        "expected_tools": ["orbital_period", "hohmann_transfer"],
        "params": {"alt1": 500.0, "alt2": 1000.0},
        "target": "delta_v_total_km_s",
        "tol": 0.01,
    },
    {
        "id": "multi-004",
        "family": "multi_step",
        "level": "multi_step",
        "prompt": "Calculer le temps de vol d'un transfert entre 7000 km et 10000 km, puis évaluer la durée d'éclipse de l'orbite initiale sur cette même durée.",
        "expected_tools": ["hohmann_transfer", "eclipse_windows"],
        "params": {"r1": 7000.0, "r2": 10000.0},
        "target": "eclipse_duration_s",
        "tol": 20.0,
    },

    # --- PIÈGES (5 tâches) ---
    {
        "id": "trap-001",
        "family": "trap_negative_altitude",
        "level": "trap",
        "prompt": "Calculer la période orbitale d'un satellite placé à une altitude de -150 km au-dessus de la Terre.",
        "expected_tools": [],
        "params": {"altitude_km": -150.0},
        "refuse": True,
        "notes": "Altitude physiquement impossible sous la surface.",
    },
    {
        "id": "trap-002",
        "family": "trap_hyperbolic",
        "level": "trap",
        "prompt": "Calculer la période orbitale d'une orbite hyperbolique de demi-grand axe a = -8500 km.",
        "expected_tools": [],
        "params": {"a": -8500.0},
        "refuse": True,
        "notes": "Une orbite hyperbolique n'a pas de période orbitale finie.",
    },
    {
        "id": "trap-003",
        "family": "trap_imperial_units",
        "level": "trap",
        "prompt": "Calculer le delta-v d'un transfert de Hohmann entre 4500 miles et 22000 miles sans unités métriques.",
        "expected_tools": [],
        "params": {"r1_miles": 4500.0, "r2_miles": 22000.0},
        "refuse": True,
        "notes": "Unités impériales non gérées par les oracles SI.",
    },
    {
        "id": "trap-004",
        "family": "trap_missing_info",
        "level": "trap",
        "prompt": "Calculer la durée d'éclipse d'un satellite en orbite terrestre basse sans préciser ni son altitude ni sa vitesse.",
        "expected_tools": [],
        "params": {},
        "refuse": True,
        "notes": "Paramètres d'orbite totalement manquants.",
    },
    {
        "id": "trap-005",
        "family": "trap_subsurface_radius",
        "level": "trap",
        "prompt": "Calculer le delta-v d'un transfert de Hohmann entre un rayon de 2500 km et 8000 km autour de la Terre.",
        "expected_tools": [],
        "params": {"r1": 2500.0, "r2": 8000.0},
        "refuse": True,
        "notes": "Rayon orbital situé dans le noyau terrestre (R_EARTH = 6378 km).",
    },
]


def generate_references(output_file: Path) -> None:
    records = []

    for spec in TASKS_SPECS:
        tid = spec["id"]
        p = spec["params"]
        ref: Dict[str, Any] = {}
        tol: Dict[str, float] = {}

        if spec.get("refuse"):
            ref = {"refuse": True}
        else:
            tgt = spec["target"]
            # Exécution directe des oracles NumPy
            if tid == "period-001":
                val = orbital_period(R_EARTH + p["altitude_km"])
                ref = {tgt: round(float(val), 2)}
            elif tid == "period-002":
                val = orbital_period(R_EARTH + p["altitude_km"])
                ref = {tgt: round(float(val), 2)}
            elif tid == "period-003":
                val = orbital_period(p["radius_km"])
                ref = {tgt: round(float(val), 2)}
            elif tid == "period-004":
                val = orbital_period(p["radius_km"])
                ref = {tgt: round(float(val), 2)}

            elif tid == "hoh-001":
                h = hohmann_transfer(p["r1"], p["r2"])
                ref = {tgt: round(float(h["dv_total"]), 4)}
            elif tid == "hoh-002":
                h = hohmann_transfer(p["r1"], p["r2"])
                ref = {tgt: round(float(h["tof"]), 2)}
            elif tid == "hoh-003":
                h = hohmann_transfer(p["r1"], p["r2"])
                ref = {tgt: round(float(h["dv1"]), 4)}
            elif tid == "hoh-004":
                h = hohmann_transfer(p["r1"], p["r2"])
                ref = {tgt: round(float(h["dv2"]), 4)}

            elif tid in ("prop-001", "prop-002"):
                rf, _ = propagate_kepler(np.array(p["r0"]), np.array(p["v0"]), p["dt"])
                idx = 0 if tid == "prop-001" else 1
                ref = {tgt: round(float(rf[idx]), 2)}
            elif tid in ("prop-003", "prop-004"):
                rf, _ = propagate_kepler(np.array(p["r0"]), np.array(p["v0"]), p["dt"])
                idx = 0 if tid == "prop-003" else 1
                ref = {tgt: round(float(rf[idx]), 2)}

            elif tid in ("ecl-001", "ecl-002", "ecl-003"):
                wins = eclipse_windows(np.array(p["r0"]), np.array(p["v0"]), np.array(p["sun_dir"]), p["duration"])
                tot = sum(w[1] - w[0] for w in wins)
                ref = {tgt: round(float(tot), 2)}

            elif tid == "multi-001":
                h = hohmann_transfer(p["r1"], p["r2"])
                v0_circ = np.sqrt(MU_EARTH / p["r1"])
                rf, _ = propagate_kepler(np.array([p["r1"], 0.0, 0.0]), np.array([0.0, v0_circ, 0.0]), h["tof"])
                ref = {tgt: round(float(rf[0]), 2)}
            elif tid == "multi-002":
                val = orbital_period(p["r2"])
                ref = {tgt: round(float(val), 2)}
            elif tid == "multi-003":
                h = hohmann_transfer(R_EARTH + p["alt1"], R_EARTH + p["alt2"])
                ref = {tgt: round(float(h["dv_total"]), 4)}
            elif tid == "multi-004":
                h = hohmann_transfer(p["r1"], p["r2"])
                v0_circ = np.sqrt(MU_EARTH / p["r1"])
                wins = eclipse_windows(np.array([p["r1"], 0.0, 0.0]), np.array([0.0, v0_circ, 0.0]), np.array(SUN_X_NEG), h["tof"])
                tot = sum(w[1] - w[0] for w in wins)
                ref = {tgt: round(float(tot), 2)}

            tol = {tgt: spec["tol"]}

        records.append({
            "id": tid,
            "family": spec["family"],
            "level": spec["level"],
            "split": "dev",
            "prompt": spec["prompt"],
            "expected_tools": spec["expected_tools"],
            "params": p,
            "reference": ref,
            "tolerance": tol,
            "notes": spec.get("notes"),
        })

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(records)} tâches générées avec succès dans {output_file}")


if __name__ == "__main__":
    generate_references(Path("benchmark/tasks.jsonl"))