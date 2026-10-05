"""Independent ground-truth checks: geometry, analytic motion and Cartesian RK4.

This module deliberately calls no scientific backend routines for its independent
answer. It never reads the
stored expected values while computing an independent answer. Formula derivations
and the integration accuracy study are documented in docs/ground-truth.md.
"""

import math

import numpy as np

from .catalog import RECIPES
from .reference import Outcome, calculate
from .schema import BenchmarkTask, BooleanResult, Failure, TaskMetadata


def _input_error(task: TaskMetadata) -> str | None:
    """Check task policy and physical inequalities independently of the backend."""
    contract = RECIPES[task.recipe]
    for name, unit in contract.inputs.items():
        if task.inputs[name].value is None:
            return "missing_input"
        if task.inputs[name].unit != unit:
            return "inconsistent_units"
    if task.context.dynamics != "two_body":
        return "unsupported_model"
    p = {name: quantity.value for name, quantity in task.inputs.items()}
    accuracy = task.context.requested_time_accuracy_s
    if accuracy is not None and "tol" in p and accuracy < p["tol"]:
        return "unachievable_precision"
    vector_names = {name for name in p if name.startswith("v") or name in ("r0", "r_sat", "sun_dir")}
    if "v1" in p:
        vector_names |= {"r1", "r2"}
    for name, value in p.items():
        if name in vector_names and (not isinstance(value, list) or len(value) != 3):
            return "invalid_input"
        if name not in vector_names and isinstance(value, list):
            return "invalid_input"
    for name in ("mu", "r_body", "step", "tol"):
        if name in p and (isinstance(p[name], list) or p[name] <= 0):
            return "invalid_input"
    for name in ("duration", "window"):
        if name in p and (isinstance(p[name], list) or p[name] < 0):
            return "invalid_input"
    if "sun_dir" in p and np.linalg.norm(p["sun_dir"]) == 0:
        return "invalid_input"
    for name in ("r0", "r1", "r2", "r_start", "r_target_a", "r_target_b", "r_sat"):
        if name in p:
            radius = np.linalg.norm(p[name]) if isinstance(p[name], list) else p[name]
            if radius <= 0:
                return "invalid_input"
            if radius < task.context.body_radius_km:
                return "physical_impossibility"
    for r_name, v_name in (("r0", "v0"), ("r1", "v1"), ("r2", "v2")):
        if v_name not in p:
            continue
        r, v = np.asarray(p[r_name]), np.asarray(p[v_name])
        radius, speed = np.linalg.norm(r), np.linalg.norm(v)
        angular_momentum = np.linalg.norm(np.cross(r, v))
        energy = np.dot(v, v) / 2 - p["mu"] / radius
        if speed == 0 or angular_momentum <= 1e-12 * radius * speed or energy >= 0:
            return "invalid_input"
        eccentricity = math.sqrt(max(0.0, 1 + 2 * energy * angular_momentum**2 / p["mu"]**2))
        # p/(1+e) computes periapsis without extracting orbital angles.
        periapsis = angular_momentum**2 / (p["mu"] * (1 + eccentricity))
        if periapsis < task.context.body_radius_km:
            return "physical_impossibility"
    return None


def integrate_cartesian(r0, v0, duration, mu, max_step=2.0):
    """Fixed-step RK4 of r'=v, v'=-mu*r/|r|³, returning position and velocity.

    Used only as a verification implementation, for the bounded time horizons
    and eccentricities of the committed corpus. A step-halving study bounds its
    numerical error; it is not a new production propagator.
    """
    state = np.concatenate((np.asarray(r0, float), np.asarray(v0, float)))
    count = max(1, math.ceil(abs(duration) / max_step))
    step = duration / count

    def derivative(value):
        position = value[:3]
        return np.concatenate((value[3:], -mu * position / np.linalg.norm(position)**3))

    for _ in range(count):
        k1 = derivative(state)
        k2 = derivative(state + step * k1 / 2)
        k3 = derivative(state + step * k2 / 2)
        k4 = derivative(state + step * k3)
        state += step * (k1 + 2*k2 + 2*k3 + k4) / 6
    return state[:3], state[3:]


def circular_state(position, velocity, time, mu):
    """Analytic circular Cartesian motion, without orbital-element conversion."""
    position, velocity = np.asarray(position), np.asarray(velocity)
    radius = np.linalg.norm(position)
    if not np.isclose(np.dot(velocity, velocity), mu / radius, rtol=1e-12, atol=0):
        raise ValueError("Analytic circular verification requires a circular state.")
    if abs(np.dot(position, velocity)) > 1e-8:
        raise ValueError("Circular verification requires perpendicular position and velocity.")
    rate = math.sqrt(mu / radius**3)
    cosine, sine = math.cos(rate * time), math.sin(rate * time)
    return position * cosine + velocity / rate * sine, velocity * cosine - position * rate * sine


def _transfer(p):
    first, second, mu = p["r1"], p["r2"], p["mu"]
    energy = -mu / (first + second)
    dv1 = abs(math.sqrt(2 * (mu / first + energy)) - math.sqrt(mu / first))
    dv2 = abs(math.sqrt(mu / second) - math.sqrt(2 * (mu / second + energy)))
    duration = math.pi / math.sqrt(mu / ((first + second) / 2)**3)
    return {"dv1": dv1, "dv2": dv2, "dv_total": dv1 + dv2, "tof": duration}


def _shadow(position, sun, body_radius):
    position, sun = np.asarray(position), np.asarray(sun)
    sun = sun / np.linalg.norm(sun)
    along = float(np.dot(position, sun))
    return bool(along < 0 and np.dot(position, position) - along**2 < body_radius**2)


def _eclipse(p):
    # In a circular orbit, Sun-axis projection is r*C*cos(n*t - phase).
    position, velocity = np.asarray(p["r0"]), np.asarray(p["v0"])
    radius = np.linalg.norm(position)
    circular_state(position, velocity, 0, p["mu"])
    sun = np.asarray(p["sun_dir"]) / np.linalg.norm(p["sun_dir"])
    first = float(np.dot(position / radius, sun))
    second = float(np.dot(velocity / np.linalg.norm(velocity), sun))
    amplitude = math.hypot(first, second)
    threshold = math.sqrt(max(0.0, 1 - (p["r_body"] / radius)**2))
    if p["duration"] == 0 or amplitude <= threshold:
        return {"total_duration": 0.0, "boundaries": []}
    rate = math.sqrt(p["mu"] / radius**3)
    phase = math.atan2(second, first)
    half_angle = math.acos(threshold / amplitude)
    period = 2 * math.pi / rate
    entry = (phase + math.pi - half_angle) / rate
    exit_ = (phase + math.pi + half_angle) / rate
    windows = []
    for lap in range(math.floor(-exit_ / period) - 1, math.ceil((p["duration"] - entry) / period) + 1):
        start = max(0.0, entry + lap * period)
        end = min(p["duration"], exit_ + lap * period)
        if start < end:
            windows.append((start, end))
    return {"total_duration": sum(end-start for start, end in windows),
            "boundaries": [boundary for interval in windows for boundary in interval]}


def _proximity(p):
    # Equal-radius circular trajectories have a separation² sinusoid at 2*n.
    first, second = np.asarray(p["r1"]), np.asarray(p["r2"])
    radius = np.linalg.norm(first)
    if not np.isclose(np.linalg.norm(second), radius, rtol=1e-12):
        raise ValueError("This analytic proximity certificate requires equal-radius circles.")
    circular_state(first, p["v1"], 0, p["mu"])
    circular_state(second, p["v2"], 0, p["mu"])
    rate = math.sqrt(p["mu"] / radius**3)
    u1, u2 = first / radius, second / radius
    w1, w2 = np.asarray(p["v1"]) / (rate * radius), np.asarray(p["v2"]) / (rate * radius)
    cosine = float(np.dot(u1, u2) - np.dot(w1, w2))
    sine = float(np.dot(u1, w2) + np.dot(w1, u2))
    candidates = [0.0, p["window"]]
    if math.hypot(cosine, sine) > 1e-12:
        phase = math.atan2(sine, cosine)
        for lap in range(-1, math.ceil(rate * p["window"] / math.pi) + 2):
            time = (phase + 2 * math.pi * lap) / (2 * rate)
            if 0 <= time <= p["window"]:
                candidates.append(time)

    def distance(time):
        a, _ = circular_state(first, p["v1"], time, p["mu"])
        b, _ = circular_state(second, p["v2"], time, p["mu"])
        return float(np.linalg.norm(a - b))

    time = min(candidates, key=lambda candidate: (distance(candidate), candidate))
    return {"tca": time, "miss_distance": distance(time)}


def independent_answer(task: TaskMetadata) -> Outcome:
    """Produce an independent answer without consulting expected results."""
    error = _input_error(task)
    if error is not None:
        return Outcome({}, error)
    p = {name: quantity.value for name, quantity in task.inputs.items()}
    recipe = task.recipe
    if recipe == "propagation":
        solver = integrate_cartesian if task.verification.method == "cartesian_rk4" else circular_state
        r, v = solver(p["r0"], p["v0"], p["dt"], p["mu"])
        values = {"position": r.tolist(), "velocity": v.tolist()}
    elif recipe == "delta_v":
        values = {"delta_v": math.sqrt(sum((b-a)**2 for a, b in zip(p["v_initial"], p["v_final"])))}
    elif recipe == "circularization":
        r, v = np.asarray(p["r0"]), np.asarray(p["v0"])
        radial_speed = float(np.dot(r, v) / np.linalg.norm(r))
        transverse_speed = np.linalg.norm(np.cross(r, v)) / np.linalg.norm(r)
        circular_speed = math.sqrt(p["mu"] / np.linalg.norm(r))
        values = {"delta_v": math.hypot(radial_speed, transverse_speed - circular_speed)}
    elif recipe == "hohmann":
        values = _transfer(p)
    elif recipe == "compare_transfers":
        common = {"r1": p["r_start"], "mu": p["mu"]}
        a = _transfer(common | {"r2": p["r_target_a"]})["dv_total"]
        b = _transfer(common | {"r2": p["r_target_b"]})["dv_total"]
        values = {"dv_a": a, "dv_b": b, "difference": b-a}
    elif recipe == "shadow":
        values = {"in_shadow": _shadow(p["r_sat"], p["sun_dir"], p["r_body"])}
    elif recipe == "eclipse":
        values = _eclipse(p)
    elif recipe == "proximity":
        values = _proximity(p)
    elif recipe == "separation":
        first, _ = circular_state(p["r1"], p["v1"], p["dt"], p["mu"])
        second, _ = circular_state(p["r2"], p["v2"], p["dt"], p["mu"])
        values = {"distance": float(np.linalg.norm(first - second))}
    elif recipe == "transfer_propagation":
        transfer = _transfer(p)
        position, _ = integrate_cartesian(p["r0"], p["v0"], transfer["tof"], p["mu"])
        values = transfer | {"position": position.tolist()}
    elif recipe == "transfer_eclipse":
        transfer = _transfer(p)
        values = {"tof": transfer["tof"], "total_duration": _eclipse(p | {"duration": transfer["tof"]})["total_duration"]}
    elif recipe == "transfer_arrival_shadow":
        transfer = _transfer(p)
        position = [-p["r2"], 0.0, 0.0]  # Antipodal arrival after half the ellipse.
        values = {"tof": transfer["tof"], "arrival_position": position,
                  "in_shadow": _shadow(position, p["sun_dir"], p["r_body"])}
    else:
        raise ValueError(f"No independent verifier for recipe {recipe}.")
    return Outcome({name: values[name] for name in task.outputs})


def verify_task(task: BenchmarkTask) -> dict[str, float]:
    """Check backend and stored reference against an independent certificate.

    Return observed absolute errors for a reviewable accuracy report. Numerical
    comparisons enforce shape, finiteness and per-output verification budgets.
    """
    backend, independent = calculate(task), independent_answer(task)
    if isinstance(task.expected, Failure):
        if backend.error != task.expected.code or independent.error != task.expected.code:
            raise ValueError(f"{task.id}: error outcome is not reproduced independently: {backend.error}, {independent.error}.")
        return {}
    if backend.error or independent.error:
        raise ValueError(f"{task.id}: expected success, got {backend.error}, {independent.error}.")
    errors = {}
    for name, expected in task.expected.outputs.items():
        if isinstance(expected, BooleanResult):
            if backend.values[name] is not expected.value or independent.values[name] is not expected.value:
                raise ValueError(f"{task.id}/{name}: Boolean certificate disagrees.")
            continue
        for actual in (backend.values[name], independent.values[name]):
            array, target = np.asarray(actual), np.asarray(expected.value)
            if array.shape != target.shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{task.id}/{name}: invalid reference shape or nonfinite result.")
        difference = float(np.max(np.abs(np.asarray(backend.values[name]) - independent.values[name])))
        stored_difference = float(np.max(np.abs(np.asarray(expected.value) - independent.values[name])))
        errors[name] = max(difference, stored_difference)
        if errors[name] > task.verification.absolute_tolerances[name]:
            raise ValueError(f"{task.id}/{name}: independent error {errors[name]} exceeds verification budget.")
    return errors
