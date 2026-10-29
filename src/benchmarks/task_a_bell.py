"""
NEXUS-OPS | QFF 2026 Phase 2
Task A: Bell-State Preparation and Fidelity Benchmark
Uses official benchmark/qff_qml_benchmark.py
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT.parent / "QFF2026_QML_Track" / "QFF2026_QML_Challenge_Kit"
BENCHMARK_DIR = KIT / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))

from qff_qml_benchmark import task_a_bell

OUT_METRICS = ROOT / "outputs" / "metrics"
OUT_METRICS.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 65)
    print("NEXUS-OPS | TASK A | BELL-STATE FIDELITY BENCHMARK")
    print("=" * 65)

    res = task_a_bell(shots=8192)

    print(f"Shots per basis:      {res['shots']}")
    print(f"Noise model version:  {res['noise_model_version']}")
    print("\n[IDEAL SIMULATION]")
    print(f"  <XX> = {res['ideal']['XX']:+.4f} +/- {res['ideal']['XX_se']:.4f}")
    print(f"  <YY> = {res['ideal']['YY']:+.4f} +/- {res['ideal']['YY_se']:.4f}")
    print(f"  <ZZ> = {res['ideal']['ZZ']:+.4f} +/- {res['ideal']['ZZ_se']:.4f}")
    print(f"  Fidelity F = {res['ideal']['fidelity']:.4f} +/- {res['ideal']['fidelity_se']:.4f}")

    print("\n[NOISY SIMULATION (QFF2026-v2)]")
    print(f"  <XX> = {res['noisy']['XX']:+.4f} +/- {res['noisy']['XX_se']:.4f}")
    print(f"  <YY> = {res['noisy']['YY']:+.4f} +/- {res['noisy']['YY_se']:.4f}")
    print(f"  <ZZ> = {res['noisy']['ZZ']:+.4f} +/- {res['noisy']['ZZ_se']:.4f}")
    print(f"  Fidelity F = {res['noisy']['fidelity']:.4f} +/- {res['noisy']['fidelity_se']:.4f}")

    out_file = OUT_METRICS / "task_a_bell.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=4)

    print(f"\nSaved Task A results to: {out_file.name}")
    print("=" * 65)


if __name__ == "__main__":
    main()
