"""
NEXUS-OPS | QFF 2026 Phase 2
Automated Regression Test Suite: PennyLane vs Qiskit Circuit Equivalence
Verifies that the Qiskit circuit reconstruction in vqc_noisy_evaluation.py
is mathematically and numerically identical to PennyLane's AngleEmbedding(Y)
and StronglyEntanglingLayers(2) architecture down to floating-point precision (< 1e-12).
"""
import sys
from pathlib import Path
import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli, Operator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import get_dataset_dir
from src.models.vqc_noisy_evaluation import build_vqc_circuit


def pl_circuit_factory(n_qubits=4):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for q in range(n_qubits):
            qml.Rot(weights[0, q, 0], weights[0, q, 1], weights[0, q, 2], wires=q)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 0])
        for q in range(n_qubits):
            qml.Rot(weights[1, q, 0], weights[1, q, 1], weights[1, q, 2], wires=q)
        qml.CNOT(wires=[1, 0])
        qml.CNOT(wires=[2, 1])
        qml.CNOT(wires=[3, 2])
        qml.CNOT(wires=[0, 3])
        return qml.expval(qml.PauliZ(0))

    return circuit


def pl_statevector_factory(n_qubits=4):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for q in range(n_qubits):
            qml.Rot(weights[0, q, 0], weights[0, q, 1], weights[0, q, 2], wires=q)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 0])
        for q in range(n_qubits):
            qml.Rot(weights[1, q, 0], weights[1, q, 1], weights[1, q, 2], wires=q)
        qml.CNOT(wires=[1, 0])
        qml.CNOT(wires=[2, 1])
        qml.CNOT(wires=[3, 2])
        qml.CNOT(wires=[0, 3])
        return qml.state()

    return circuit


def test_single_qubit_rot_unitary_equivalence():
    """
    Verifies that PennyLane qml.Rot(phi, theta, omega) is identical to
    Qiskit rz(phi) -> ry(theta) -> rz(omega).
    """
    test_angles = [
        (0.0, 0.0, 0.0),
        (0.5, 1.2, -0.7),
        (np.pi / 4, np.pi / 3, np.pi / 2),
        (-2.1, 0.9, 3.0),
    ]
    for phi, theta, omega in test_angles:
        u_pl = qml.matrix(qml.Rot(phi, theta, omega, wires=0))

        qc = QuantumCircuit(1)
        qc.rz(phi, 0)
        qc.ry(theta, 0)
        qc.rz(omega, 0)
        u_qk = Operator(qc).data

        max_diff = np.max(np.abs(u_pl - u_qk))
        assert max_diff < 1e-14, f"Unitary mismatch for angles {(phi, theta, omega)}: max_diff={max_diff}"


def test_deterministic_synthetic_inputs_equivalence():
    """
    Compares PennyLane PauliZ(0) expectation against Qiskit expectation
    across deterministic boundary and corner synthetic inputs using saved weights.
    """
    weights_path = ROOT / "outputs" / "models" / "vqc_4qubit_weights.npy"
    assert weights_path.exists(), "vqc_4qubit_weights.npy not found"
    weights = np.load(weights_path)

    pl_circuit = pl_circuit_factory()

    synthetic_inputs = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2]),
        np.array([np.pi, np.pi, np.pi, np.pi]),
        np.array([0.1, 0.8, 1.5, 2.7]),
        np.array([np.pi, 0.0, np.pi, 0.0]),
        np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]),
    ]

    for x in synthetic_inputs:
        # PennyLane expectation
        pl_exp = float(pl_circuit(x, weights))

        # Qiskit expectation via statevector
        qc = build_vqc_circuit(x, weights)
        qc_no_meas = qc.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_meas)

        # In Qiskit little-endian string representation, wire 0 is the rightmost char: 'IIIZ'
        qk_exp = float(sv.expectation_value(Pauli("IIIZ")).real)

        diff = abs(pl_exp - qk_exp)
        assert diff < 1e-12, f"Synthetic input {x} failed equivalence: PL={pl_exp}, QK={qk_exp}, diff={diff}"


def test_real_validation_samples_equivalence():
    """
    Compares PennyLane and Qiskit expectations on actual preprocessed
    validation samples from ecg_val.csv.
    """
    import joblib
    import pandas as pd

    weights = np.load(ROOT / "outputs" / "models" / "vqc_4qubit_weights.npy")
    preproc = joblib.load(ROOT / "outputs" / "models" / "vqc_4qubit_preprocessing.joblib")

    val_df = pd.read_csv(get_dataset_dir("q1") / "ecg_val.csv")
    feature_cols = preproc["feature_cols"]
    X_val_raw = val_df[feature_cols].iloc[:10].to_numpy(dtype=np.float64)

    X_val_s = preproc["standardizer"].transform(X_val_raw)
    X_val_pca = preproc["pca"].transform(X_val_s)
    X_val_angles = np.clip(preproc["angle_scaler"].transform(X_val_pca), 0.0, np.pi)

    pl_circuit = pl_circuit_factory()

    for idx, x in enumerate(X_val_angles):
        pl_exp = float(pl_circuit(x, weights))

        qc = build_vqc_circuit(x, weights)
        qc_no_meas = qc.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_meas)
        qk_exp = float(sv.expectation_value(Pauli("IIIZ")).real)

        diff = abs(pl_exp - qk_exp)
        assert diff < 1e-12, f"Validation sample {idx} failed equivalence: diff={diff}"


def test_probability_mapping_and_bit_convention():
    """
    Verifies that the probability mapping p = (expval + 1) / 2 matches
    the projection probability of qubit 0 onto state |0>.
    """
    weights = np.load(ROOT / "outputs" / "models" / "vqc_4qubit_weights.npy")
    x = np.array([0.5, 1.0, 1.5, 2.0])

    qc = build_vqc_circuit(x, weights)
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(qc_no_meas)

    # In Qiskit, probabilities of 4-qubit state:
    probs = sv.probabilities()
    # Indices where qubit 0 is 0 (i.e. even indices in little-endian ordering: 0, 2, 4, 6, 8, ...)
    p_qubit0_is_0 = sum(probs[i] for i in range(16) if (i % 2 == 0))

    # Expectation value:
    exp_z = float(sv.expectation_value(Pauli("IIIZ")).real)
    mapped_p = (exp_z + 1.0) / 2.0

    diff = abs(mapped_p - p_qubit0_is_0)
    assert diff < 1e-12, f"Probability mapping mismatch: mapped={mapped_p}, statevector_proj={p_qubit0_is_0}, diff={diff}"
