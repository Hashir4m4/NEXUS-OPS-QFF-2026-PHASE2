"""
NEXUS-OPS | QFF 2026 Phase 2
Task B: Processor D Design, Benchmark, and Validation
Uses official benchmark/qff_qml_benchmark.py
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT.parent / "QFF2026_QML_Track" / "QFF2026_QML_Challenge_Kit"
BENCHMARK_DIR = KIT / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))

from qff_qml_benchmark import validate_processor_d, task_b_qft3

CONFIG_DIR = ROOT / "configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 65)
    print("NEXUS-OPS | TASK B | PROCESSOR D BENCHMARK")
    print("=" * 65)

    processor_d = {
        "processor_id": "nexus_ops_ring4_v1",
        "n_qubits": 4,
        "coupling_map": [
            [0, 1], [1, 0],
            [1, 2], [2, 1],
            [2, 3], [3, 2],
            [3, 0], [0, 3]
        ],
        "native_gate_set": [
            "rx",
            "ry",
            "rz",
            "cx"
        ],
        "qubit_layout": "ring",
        "noise_assumptions": {
            "single_qubit_error_rate": 0.001,
            "two_qubit_error_rate": 0.01,
            "readout_error_rate": 0.015,
            "t1_us": 120.0,
            "t2_us": 90.0,
            "notes": "Superconducting transmon processor with nearest-neighbor microwave resonators arranged in a 4-qubit planar ring."
        },
        "design_rationale": "Processor D is engineered as a 4-qubit bidirectional ring tailored to the NEXUS-OPS PS-Q1 biomedical signal classification pipeline. The 4 physical qubits directly correspond to our 4 principal component ECG features, retaining 81.3% of cumulative variance without wasted physical overhead. The closed ring topology matches the periodic nearest-neighbor entangling topology of StronglyEntanglingLayers, guaranteeing zero-SWAP overhead for all circular entangling operations in our classifier. The inclusion of bidirectional CX links eliminates the 4-Hadamard reversal cost (avoiding single-qubit error accumulation of p=0.01 per gate), and the low degree (d=2) minimizes inter-qubit frequency crowding and ZZ-crosstalk. For universal benchmark workloads like QFT-3, this topology incurs only a single routing SWAP while preserving high two-qubit execution fidelity."
    }

    pre_errors = validate_processor_d(processor_d, require_results=False)
    if pre_errors:
        print("Design validation failed before benchmark:")
        for err in pre_errors:
            print(f"  - {err}")
        sys.exit(1)

    print("Running official Task B QFT-3 benchmark on Processor D...")
    benchmark_res = task_b_qft3(processor_d)
    processor_d["benchmark_results"] = benchmark_res

    print("\nOfficial QFT-3 Benchmark Results:")
    print(json.dumps(benchmark_res, indent=2))

    final_errors = validate_processor_d(processor_d, require_results=True)
    if final_errors:
        print("Final validation failed:")
        for err in final_errors:
            print(f"  - {err}")
        sys.exit(1)

    out_file = CONFIG_DIR / "processor_d_config.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(processor_d, f, indent=2)

    print(f"\n[PASS] Processor D validated and saved to: {out_file.name}")
    print("=" * 65)


if __name__ == "__main__":
    main()

