"""
NEXUS-OPS | QFF 2026 Phase 2
Comprehensive Zero-dependency Test Runner
Executes all unit, behavioral, equivalence, and benchmark validation test suites.
"""
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import get_tools_dir, get_benchmark_dir
sys.path.insert(0, str(get_tools_dir()))
sys.path.insert(0, str(get_benchmark_dir()))

# Import all individual test suites
from tests.test_dataset import (
    test_dataset_files_exist,
    test_dataset_dimensions,
    test_column_consistency,
    test_no_missing_values,
    test_label_validity,
    test_test_sample_ids
)
from tests.test_leakage_and_preprocessing import (
    test_preprocessing_artifacts_exist,
    test_no_leakage_scaler_and_pca,
    test_angle_scaler_bounds,
    test_preprocessing_reproducibility
)
from tests.test_models import (
    test_model_files_exist,
    test_vqc_weights_dimension,
    test_vqc_validation_probabilities_bound,
    test_vqc_performance_thresholds,
    test_vqc_live_circuit_execution_consistency
)
from tests.test_circuit_equivalence import (
    test_single_qubit_rot_unitary_equivalence,
    test_deterministic_synthetic_inputs_equivalence,
    test_real_validation_samples_equivalence,
    test_probability_mapping_and_bit_convention
)
from tests.test_task_a import (
    test_task_a_artifact_exists,
    test_task_a_metrics
)
from tests.test_task_b import (
    test_processor_d_file_exists,
    test_processor_d_schema_and_validation,
    test_processor_d_properties
)
from tests.test_predictions import (
    test_prediction_file_exists,
    test_official_prediction_validator,
    test_prediction_rows_and_format
)
from tests.test_missing_resources import (
    test_invalid_env_var_kit_path,
    test_invalid_problem_dataset_dir,
    test_validator_clean_failure_on_missing_file,
    test_processor_d_clean_failure_on_corrupt_config
)


def run_all_suites():
    t_start = time.time()
    print("=" * 70)
    print("NEXUS-OPS | COMPREHENSIVE AUTOMATED VERIFICATION SUITE")
    print("=" * 70)

    suites = [
        ("1. Dataset Integrity & Schema Checks", [
            test_dataset_files_exist,
            test_dataset_dimensions,
            test_column_consistency,
            test_no_missing_values,
            test_label_validity,
            test_test_sample_ids
        ]),
        ("2. Data Leakage & Preprocessing Verification", [
            test_preprocessing_artifacts_exist,
            test_no_leakage_scaler_and_pca,
            test_angle_scaler_bounds,
            test_preprocessing_reproducibility
        ]),
        ("3. Quantum Model Artifacts & Behavioral Inference", [
            test_model_files_exist,
            test_vqc_weights_dimension,
            test_vqc_validation_probabilities_bound,
            test_vqc_performance_thresholds,
            test_vqc_live_circuit_execution_consistency
        ]),
        ("4. PennyLane & Qiskit Circuit Equivalence", [
            test_single_qubit_rot_unitary_equivalence,
            test_deterministic_synthetic_inputs_equivalence,
            test_real_validation_samples_equivalence,
            test_probability_mapping_and_bit_convention
        ]),
        ("5. Task A Bell-State Fidelity Benchmark", [
            test_task_a_artifact_exists,
            test_task_a_metrics
        ]),
        ("6. Task B Processor D Schema & QFT-3 Benchmark", [
            test_processor_d_file_exists,
            test_processor_d_schema_and_validation,
            test_processor_d_properties
        ]),
        ("7. Official Prediction File & Validator Test", [
            test_prediction_file_exists,
            test_official_prediction_validator,
            test_prediction_rows_and_format
        ]),
        ("8. Missing Resources & Error Handling", [
            test_invalid_env_var_kit_path,
            test_invalid_problem_dataset_dir,
            test_validator_clean_failure_on_missing_file,
            test_processor_d_clean_failure_on_corrupt_config
        ])
    ]

    total_suites = len(suites)
    passed_suites = 0
    total_assertions = 0

    for suite_name, test_fns in suites:
        suite_t0 = time.time()
        suite_passed = True
        err_msg = ""
        for fn in test_fns:
            try:
                fn()
                total_assertions += 1
            except Exception as e:
                suite_passed = False
                err_msg = f"{fn.__name__} failed: {e}"
                break

        dur = time.time() - suite_t0
        if suite_passed:
            print(f"[PASS] {suite_name:<50} ({len(test_fns)} checks, {dur:.2f}s)")
            passed_suites += 1
        else:
            print(f"[FAIL] {suite_name:<50} -> {err_msg}")

    total_dur = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"VERIFICATION RESULTS: {passed_suites}/{total_suites} SUITES PASSED ({total_assertions} checks passed in {total_dur:.2f}s)")
    print("=" * 70)

    if passed_suites != total_suites:
        sys.exit(1)


if __name__ == "__main__":
    run_all_suites()
