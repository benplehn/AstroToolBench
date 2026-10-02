import json
import pytest
from atb.executor import execute_tool


def test_valid_call_version_b():
    raw_res = execute_tool("B", "hohmann_transfer", '{"r1": 7000.0, "r2": 42164.0}')
    data = json.loads(raw_res)
    assert "dv_total" in data
    assert pytest.approx(data["dv_total"], rel=1e-3) == 3.771


def test_valid_call_version_c():
    # 621.863 km alt + R_EARTH (6378.137) = 7000 km radius
    raw_res = execute_tool("C", "hohmann_transfer", '{"initial_altitude_km": 621.863, "final_altitude_km": 35785.863}')
    data = json.loads(raw_res)
    assert "delta_v_total_km_s" in data
    assert pytest.approx(data["delta_v_total_km_s"], rel=1e-3) == 3.771


def test_unknown_tool():
    res_b = json.loads(execute_tool("B", "teleport_orbit", '{}'))
    assert "Unknown tool" in res_b["error"]

    res_c = json.loads(execute_tool("C", "teleport_orbit", '{}'))
    assert "is not recognized" in res_c["error"]


def test_malformed_json():
    res_b = json.loads(execute_tool("B", "orbital_period", '{bad json'))
    assert "error" in res_b
    assert "JSONDecodeError" in res_b["error"]

    res_c = json.loads(execute_tool("C", "orbital_period", '{bad json'))
    assert "Malformed arguments JSON" in res_c["error"]


def test_invalid_arguments_guard_c():
    res = json.loads(execute_tool("C", "orbital_period", '{"altitude_km": -50.0}'))
    assert "error" in res
    assert "Altitude must be >= 0" in res["error"]