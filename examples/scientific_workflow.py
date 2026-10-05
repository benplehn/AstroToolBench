"""Apply a Hohmann departure impulse, propagate to arrival and check shadow.

Run from the checkout with the package installed:
    python examples/scientific_workflow.py
"""

import json
import math

from astrotoolbench.tools import MU_EARTH, hohmann_transfer, in_cylindrical_shadow, propagate_kepler


def transfer_arrival() -> dict:
    departure_radius_km, arrival_radius_km = 7000.0, 14000.0
    transfer = hohmann_transfer(departure_radius_km, arrival_radius_km)
    departure_speed_km_s = math.sqrt(MU_EARTH / departure_radius_km) + transfer["dv1"]
    position_km, velocity_km_s = propagate_kepler(
        [departure_radius_km, 0.0, 0.0], [0.0, departure_speed_km_s, 0.0], transfer["tof"],
    )
    return {
        "transfer": transfer,
        "arrival_position_km": position_km.tolist(),
        "arrival_velocity_km_s": velocity_km_s.tolist(),
        "arrival_in_shadow": in_cylindrical_shadow(position_km, [1.0, 0.0, 0.0]),
    }


if __name__ == "__main__":
    print(json.dumps(transfer_arrival(), indent=2, allow_nan=False))
