"""
NEXUS-OPS | QFF 2026 Phase 2
Automated Test Suite: Graceful Failure on Missing Resources or Invalid Inputs
Verifies requirement 11: System fails cleanly with informative error messages
when required files, directories, or challenge kit paths are missing or malformed.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import get_challenge_kit_path, get_dataset_dir, get_tools_dir, get_benchmark_dir
sys.path.insert(0, str(get_tools_dir()))
sys.path.insert(0, str(get_benchmark_dir()))

from validate_predictions import validate
from qff_qml_benchmark import validate_processor_d


def test_invalid_env_var_kit_path():
    """
    Verifies that setting QFF_CHALLENGE_KIT to a non-existent directory
    falls back or raises clean FileNotFoundError rather than crashing unhandled.
    """
    old_env = os.environ.get("QFF_CHALLENGE_KIT")
    try:
        os.environ["QFF_CHALLENGE_KIT"] = str(ROOT / "non_existent_kit_directory_12345")
        # Should fall back to valid sibling or fail cleanly
        kit = get_challenge_kit_path()
        assert kit.exists(), "Fallback kit must exist"
    finally:
        if old_env is not None:
            os.environ["QFF_CHALLENGE_KIT"] = old_env
        else:
            os.environ.pop("QFF_CHALLENGE_KIT", None)


def test_invalid_problem_dataset_dir():
    """
    Verifies that requesting a non-existent problem ID raises FileNotFoundError.
    """
    try:
        get_dataset_dir("q999_invalid")
        assert False, "Should have raised FileNotFoundError for invalid problem"
    except FileNotFoundError as e:
        assert "Dataset directory not found" in str(e)


def test_validator_clean_failure_on_missing_file():
    """
    Verifies that the official prediction validator returns clean error list
    when the target file does not exist, rather than throwing uncaught exception.
    """
    missing_file = str(ROOT / "non_existent_predictions.csv")
    errs = validate("q1", missing_file)
    assert len(errs) > 0, "Validator should return errors for missing file"
    assert any("cannot read CSV" in err for err in errs)


def test_processor_d_clean_failure_on_corrupt_config():
    """
    Verifies that validate_processor_d catches corrupt or missing fields.
    """
    corrupt_cfg = {
        "processor_id": "corrupt_test",
        "n_qubits": 2,  # Invalid: requires at least 3 qubits
        "coupling_map": [[0, 1]],
        "native_gate_set": ["rx"]  # Missing 2-qubit gate cx/cz
    }
    errs = validate_processor_d(corrupt_cfg, require_results=False)
    assert len(errs) > 0, "validate_processor_d should catch corrupt config"
