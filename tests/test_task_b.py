"""
NEXUS-OPS | QFF 2026 Phase 2
Automated Test Suite: Task B Processor D Configuration and QFT-3 Benchmark
"""
from pathlib import Path
import json
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT.parent / "QFF2026_QML_Track" / "QFF2026_QML_Challenge_Kit"
sys.path.insert(0, str(KIT / "benchmark"))

from qff_qml_benchmark import validate_processor_d

CONFIG_FILE = ROOT / "configs" / "processor_d_config.json"


def test_processor_d_file_exists():
    assert CONFIG_FILE.exists(), "processor_d_config.json missing"


def test_processor_d_schema_and_validation():
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)

    errors = validate_processor_d(cfg, require_results=True)
    assert len(errors) == 0, f"Processor D validation failed: {errors}"


def test_processor_d_properties():
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)

    assert 3 <= cfg["n_qubits"] <= 9, "n_qubits outside [3, 9]"
    assert "cx" in cfg["native_gate_set"] or "cz" in cfg["native_gate_set"]
    assert cfg["qubit_layout"] in ["linear", "ring", "star", "grid", "heavy_hex", "custom"]
    assert len(cfg["design_rationale"]) >= 50

    b = cfg["benchmark_results"]
    assert b["qft3_fidelity_ideal"] == 1.0, f"Ideal QFT-3 fidelity must be 1.0, got {b['qft3_fidelity_ideal']}"
    assert 0.5 <= b["qft3_fidelity_noisy"] <= 0.8, f"Noisy QFT-3 fidelity unexpected: {b['qft3_fidelity_noisy']}"
    assert b["benchmark_noise_model_version"] == "QFF2026-v2"

