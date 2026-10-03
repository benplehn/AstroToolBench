"""Tool schemas for OpenAI function calling (TOOLS_B: raw, TOOLS_C: agent-optimized)."""

TOOLS_B = [
    {
        "type": "function",
        "function": {
            "name": "orbital_period",
            "description": "Compute orbital period.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "Semi-major axis"},
                    "mu": {"type": "number", "description": "Gravitational parameter"}
                },
                "required": ["a"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "hohmann_transfer",
            "description": "Compute Hohmann transfer delta-v and time of flight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "r1": {"type": "number", "description": "Initial orbit radius"},
                    "r2": {"type": "number", "description": "Final orbit radius"},
                    "mu": {"type": "number", "description": "Gravitational parameter"}
                },
                "required": ["r1", "r2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "propagate_orbit",
            "description": "Propagate orbit state vector over delta t.",
            "parameters": {
                "type": "object",
                "properties": {
                    "r0": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Initial position vector"
                    },
                    "v0": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Initial velocity vector"
                    },
                    "dt": {"type": "number", "description": "Propagation time step"},
                    "mu": {"type": "number", "description": "Gravitational parameter"}
                },
                "required": ["r0", "v0", "dt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "eclipse_windows",
            "description": "Find eclipse entry and exit times in cylindrical shadow.",
            "parameters": {
                "type": "object",
                "properties": {
                    "r0": {"type": "array", "items": {"type": "number"}},
                    "v0": {"type": "array", "items": {"type": "number"}},
                    "sun_dir": {"type": "array", "items": {"type": "number"}},
                    "duration": {"type": "number"}
                },
                "required": ["r0", "v0", "sun_dir", "duration"]
            }
        }
    }
]

TOOLS_C = [
    {
        "type": "function",
        "function": {
            "name": "orbital_period",
            "description": (
                "Compute the orbital period of a circular Earth orbit. "
                "Takes altitude above Earth's surface (km). Returns orbital period in seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "altitude_km": {
                        "type": "number",
                        "minimum": 0.0,
                        "description": "Altitude above Earth's surface in kilometers (not orbital radius)."
                    }
                },
                "required": ["altitude_km"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "hohmann_transfer",
            "description": (
                "Compute coplanar circular-orbit Hohmann transfer impulses and duration. "
                "Inputs are initial and final altitudes above Earth's surface in km. "
                "Returns delta_v1_km_s, delta_v2_km_s, delta_v_total_km_s, and transfer_time_s."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "initial_altitude_km": {
                        "type": "number",
                        "minimum": 0.0,
                        "description": "Departure orbit altitude in kilometers above Earth surface (h1 >= 0)."
                    },
                    "final_altitude_km": {
                        "type": "number",
                        "minimum": 0.0,
                        "description": "Target orbit altitude in kilometers above Earth surface (h2 >= 0)."
                    }
                },
                "required": ["initial_altitude_km", "final_altitude_km"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "propagate_orbit",
            "description": (
                "Propagate a satellite Keplerian state vector in Earth-Centered Inertial (ECI) frame. "
                "Takes initial position r0_km [x,y,z], velocity v0_km_s [vx,vy,vz], and propagation duration dt_s (seconds). "
                "Returns final position r_final_km [x,y,z] and final velocity v_final_km_s [vx,vy,vz]."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "r0_km": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Initial ECI position vector [x, y, z] in km from Earth center."
                    },
                    "v0_km_s": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Initial ECI velocity vector [vx, vy, vz] in km/s."
                    },
                    "dt_s": {
                        "type": "number",
                        "description": "Propagation elapsed time in seconds (can be positive or negative)."
                    }
                },
                "required": ["r0_km", "v0_km_s", "dt_s"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "eclipse_windows",
            "description": (
                "Compute cylindrical Earth-shadow eclipse entry and exit intervals for an orbit. "
                "Takes initial state, unit sun direction vector, and analysis window duration in seconds. "
                "Returns list of eclipse intervals [start_s, end_s] and total eclipse duration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "r0_km": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Initial ECI position [x, y, z] in km."
                    },
                    "v0_km_s": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Initial ECI velocity [vx, vy, vz] in km/s."
                    },
                    "sun_dir": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Unit vector pointing from Earth toward Sun in ECI."
                    },
                    "duration_s": {
                        "type": "number",
                        "minimum": 0.0,
                        "description": "Total observation window duration in seconds."
                    }
                },
                "required": ["r0_km", "v0_km_s", "sun_dir", "duration_s"]
            }
        }
    }
]
