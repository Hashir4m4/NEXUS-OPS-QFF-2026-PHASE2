# QFF2026 QML Track — Results Report

**Team name:** NEXUS-OPS  
**Selected PS:** [x] PS-Q1  [ ] PS-Q2  [ ] PS-Q3  
**Date:** October 29, 2026  
**Dependency versions:**
```text
numpy==2.4.6
pandas==3.1.0
pennylane==0.44.1
PennyLane-qiskit==0.44.1
pennylane_lightning==0.45.0
qiskit==2.3.0
qiskit_aer==0.17.2
scikit_learn==1.9.1
```
**Noise model version:** QFF2026-v2  

---

## 1. Approach Summary

NEXUS-OPS tackles PS-Q1 (Biomedical ECG Signal Classification) using a hybrid quantum–classical pipeline designed for robustness, NISQ compliance, and clinical interpretability. The 30 morphological and statistical heartbeat features (derived from 5 segments of 35 samples around the R-peak) are standardized and projected onto the top 4 principal components fitted strictly on training data (`ecg_train.csv`), capturing 81.28% of total variance while eliminating noise dimensions. These 4 features are angle-encoded onto a 4-qubit quantum register via $R_y(\theta)$ rotations ($\theta \in [0, \pi]$), followed by a 2-layer `StronglyEntanglingLayers` variational ansatz containing 24 trainable Euler rotation parameters and periodic CNOT entanglers. Classification is performed by measuring the single-qubit expectation value $\langle Z_0 \rangle \in [-1, 1]$, mapped monotonically to the probability of Arrhythmia $P(\text{Arrhythmia}) = (\langle Z_0 \rangle + 1) / 2$. This architecture balances expressive representation with shallow circuit depth (depth 13 transpiled), ensuring noise resilience against depolarizing and readout errors.

---

## 2. Results

### Classical baseline
Official metric: Macro-averaged F1-score on `ecg_val.csv` (1,500 samples: 899 Normal, 601 Arrhythmia). Preprocessing fitted strictly on `ecg_train.csv`.

| Model | Official metric (Macro-F1) | Accuracy | Training Time | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | **1.0000** | **1.0000** | **1.155 s** | Standardized features, L-BFGS, max_iter=2000, seed=42. Perfect separability on validation set. |
| Linear SVM | 1.0000 | 1.0000 | 0.940 s | Linear kernel, C=1.0, seed=42. |
| RBF SVM | 1.0000 | 1.0000 | 1.480 s | Non-linear radial basis kernel, C=1.0, seed=42. |
| Random Forest | 1.0000 | 1.0000 | 0.820 s | 100 estimators, max_depth=None, seed=42. |
| Logistic Regression (4 PCA components) | 1.0000 | 1.0000 | 0.050 s | Demonstrates that the 4-dimensional PCA subspace retains 100% linear separability. |

### Quantum / hybrid model
Official metric: Macro-averaged F1-score on `ecg_val.csv`.

| Configuration | Ideal | Noisy (benchmark) | Notes |
| :--- | :--- | :--- | :--- |
| **4-Qubit VQC (2 Layers, 24 params)** | **0.9979** | **0.9965** | Trained on 6,000 samples for 10 epochs (Adam, lr=0.03, batch=64). Ideal: 0 FP, 3 FN out of 1,500. Noisy: 0 FP, 5 FN under QFF2026-v2 (8192 shots). |

#### Confusion Matrix Comparison on `ecg_val.csv` (1,500 beats):
- **Ideal Simulation:**
  - Normal (Class 0): TN = 899, FP = 0 (Precision = 0.9967, Recall = 1.0000)
  - Arrhythmia (Class 1): FN = 3, TP = 598 (Precision = 1.0000, Recall = 0.9950)
  - Accuracy: **0.9980** | Macro-F1: **0.9979**
- **Noisy Simulation (QFF2026-v2, 8192 shots):**
  - Normal (Class 0): TN = 899, FP = 0 (Precision = 0.9945, Recall = 1.0000)
  - Arrhythmia (Class 1): FN = 5, TP = 596 (Precision = 1.0000, Recall = 0.9917)
  - Accuracy: **0.9967** | Macro-F1: **0.9965**
  - Difference: Only $\Delta \text{Macro-F1} = -0.0014$ degradation.

### Noise mitigation (PS-Q3 required; optional for PS-Q1/Q2)
| Strategy | Mitigated metric | Extra circuit evals | Justified? |
| :--- | :--- | :--- | :--- |
| Readout Threshold Calibration & Symmetry Selection | 0.9972 | 0 (post-processing calibration) | Yes; mitigates symmetric readout bit-flip bias ($p=0.02$) near boundary. |

---

## 3. Task A — Bell state |Φ+⟩ (required)

Produced by `task_a_bell()` in `benchmark/qff_qml_benchmark.py` (8192 shots per basis).

**Fidelity definition used:** $F = (1 + \langle XX \rangle - \langle YY \rangle + \langle ZZ \rangle) / 4$, with $\langle P \rangle = (N_{\text{even}} - N_{\text{odd}})/N$, $\text{SE}(\langle P \rangle) = \sqrt{(1 - \langle P \rangle^2)/N}$ and $\text{SE}(F) = \sqrt{\text{SE}_{XX}^2 + \text{SE}_{YY}^2 + \text{SE}_{ZZ}^2}/4$. ($P(00)+P(11)$ is a population, not a fidelity, and is not accepted.)

| Condition | ⟨XX⟩ ± SE | ⟨YY⟩ ± SE | ⟨ZZ⟩ ± SE | F ± SE |
| :--- | :--- | :--- | :--- | :--- |
| **Ideal** | +1.0000 ± 0.0000 | -1.0000 ± 0.0000 | +1.0000 ± 0.0000 | **1.0000 ± 0.0000** |
| **Noisy (benchmark)** | +0.8191 ± 0.0063 | -0.8279 ± 0.0062 | +0.8735 ± 0.0054 | **0.8801 ± 0.0026** |

### Noise and Entanglement Interpretation (3–5 sentences):
Under the organizer benchmark noise model (`QFF2026-v2`), the Bell-state fidelity drops from the ideal $1.0000$ to $0.8801 \pm 0.0026$, in exact concordance with theoretical noise compounding across the preparation circuit ($H + CX$). The two-qubit depolarizing error ($p_2 = 0.05$) contracts the state towards the maximally mixed state $I/4$, reducing both diagonal coherence ($\langle ZZ \rangle = 0.8735$) and off-diagonal superposition coherences ($\langle XX \rangle = 0.8191, \langle YY \rangle = -0.8279$). Notably, $\langle XX \rangle$ and $\langle YY \rangle$ exhibit greater degradation than $\langle ZZ \rangle$ because the measurement basis rotations ($H$ and $S^\dagger H$) introduce additional single-qubit depolarizing errors ($p_1 = 0.01$). Finally, the symmetric readout bit-flip error ($p_{\text{readout}} = 0.02$) attenuates all measured parity expectations by a factor of $(1 - 2p_{\text{readout}})^2 \approx 0.9216$, showing that measurement fidelity is as critical to quantum state tomography as gate fidelity.

---

## 4. Processor D Benchmark (Task B)

Produced by `python benchmark/qff_qml_benchmark.py configs/processor_d_config.json`.

**QFT-3 fidelity definition used:** mean over the 8 computational-basis inputs $|j\rangle$ of $F_j = \langle \psi_j | \rho_j | \psi_j \rangle$, where $\psi_j = \text{QFT}|j\rangle$ (ideal) and $\rho_j$ is the density matrix of the 3 logical qubits after the transpiled circuit runs under the benchmark noise model (gate noise; no measurement). Ideal must be 1.0.

**Processor ID:** `nexus_ops_ring4_v1`  
**n_qubits:** 4  
**Topology:** `ring` (bidirectional 4-qubit cyclic loop)

| Benchmark | Value |
| :--- | :--- |
| QFT-3 depth (before transpile) | 6 |
| QFT-3 depth (after transpile) | 21 |
| Two-qubit gate count | 12 |
| SWAP overhead | 1 |
| Fidelity (ideal) | 1.000000 |
| Fidelity (noisy) | 0.626784 |
| Coupling map direction | Bidirectional (all 4 physical edges listed in both directions) |

### Topology Analysis & Rationale (2–4 sentences):
The 4-qubit bidirectional ring provides optimal topological alignment with our PS-Q1 classification circuit: the 4 physical qubits match the 4 PCA components, while the periodic nearest-neighbor links map the cyclic CNOT pattern of `StronglyEntanglingLayers` with exactly zero SWAP overhead. Bidirectional links eliminate the 4-Hadamard reversal cost on every reversed two-qubit gate, avoiding $4 \times 0.01$ depolarizing error accumulation per CX. For the non-local QFT-3 benchmark, the ring structure provides multiple routing paths, requiring only a single routing SWAP (3 CX gates) to bridge non-adjacent qubits, resulting in an ideal fidelity of 1.0 and a noisy fidelity of 0.6268 under the severe $p_2=0.05$ two-qubit depolarizing regime.

---

## 5. Resource Costs

| Resource | Value |
| :--- | :--- |
| **Qubits used** | 4 |
| **Circuit depth (raw ansatz)** | 8 (4 single-qubit rotation steps + 4 two-qubit entangling steps) |
| **Circuit depth (transpiled to rx, ry, rz, cx)** | 13 |
| **Two-qubit gates (raw)** | 8 CNOTs (4 per layer) |
| **Two-qubit gates (transpiled)** | 8 CNOTs |
| **Trainable parameters** | 24 ($2 \text{ layers} \times 4 \text{ qubits} \times 3 \text{ Euler angles}$) |
| **Training time (approx)** | 40.08 seconds (10 epochs, 94 batches/epoch, Adam lr=0.03) |
| **Total circuit evaluations (training)** | 940 batch evaluations ($\approx 60,000$ sample evaluations on statevector simulator) |
| **Total circuit evaluations (noisy validation)** | 1,500 circuits transpiled and simulated on AerSimulator ($8,192$ shots/circuit = 12,288,000 total shots) |

---

## 6. Limitations and Honest Assessment

1. **Separability vs. Quantum Advantage:**  
   The PS-Q1 dataset (derived from 30 summary statistics of MIT-BIH segments) is linearly separable in both its original 30-dimensional space and in the 4-dimensional PCA projection. Consequently, a classical Logistic Regression baseline attains a Macro-F1 of 1.0000 in 1.15 seconds. While our 4-qubit VQC achieves a competitive Macro-F1 of 0.9979 (ideal) and 0.9965 (noisy), there is no empirical quantum supremacy on this specific feature-engineered dataset. Rather, the quantum model demonstrates exceptional inductive bias and noise robustness, learning a faithful non-linear decision boundary with only 24 parameters.

2. **Dimensionality Reduction Trade-off:**  
   Compressing 30 features into 4 PCA dimensions discards 18.72% of the original variance. While this retained subspace contains complete class separation for Arrhythmia detection, subtle clinical morphologic features (e.g. minor ST-segment variations) present in higher-order PCs are sacrificed to fit the 4-qubit NISQ constraint.

3. **Simulation vs. Real Physical Hardware:**  
   All noisy experiments were executed via Qiskit Aer simulation using the organizers' `QFF2026-v2` noise model. While the model rigorously captures depolarizing and readout noise, physical superconducting hardware exhibits time-dependent parameter drift, non-Markovian $1/f$ flux noise, spectator qubit crosstalk, and thermal relaxation ($T_1$) effects that may further degrade deep circuits.

---

## 7. Reproducibility

```python
# Fixed seeds across the entire pipeline
SEED = 42
np.random.seed(42)

# Reproduction commands:
# 1. Classical baseline:
#    python src/benchmarks/classical_baseline.py
# 2. VQC Training (Ideal):
#    python src/models/vqc_4qubit.py
# 3. Task A (Bell State):
#    python src/benchmarks/task_a_bell.py
# 4. Task B (Processor D):
#    python src/benchmarks/task_b_processor_d.py
# 5. Noisy Evaluation:
#    python src/models/vqc_noisy_evaluation.py
# 6. Final Predictions:
#    python src/models/generate_predictions.py
# 7. Complete Test Suite:
#    pytest tests/ -v
```

---

## 8. Optional Extensions & Ablation Studies

### Ablation Study: Layer Count & Qubit Dimension Impact
Conducted on identical training/validation splits using 5 epochs per variant:

| Configuration | Qubits | Layers | Parameters | PCA Variance | Accuracy | Macro-F1 | Train Time | Scientific Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1-Layer VQC** | 4 | 1 | 12 | 81.28% | 0.6713 | 0.5447 | 15.09 s | Underparameterized. 1 entangling layer cannot express the required separating hyperplane. |
| **2-Layer VQC (Baseline)** | 4 | 2 | 24 | 81.28% | 0.9960 | 0.9958 | 34.64 s | Optimal balance. Expressive enough for 99.8% accuracy while maintaining shallow depth (13). |
| **3-Layer VQC** | 4 | 3 | 36 | 81.28% | 1.0000 | 1.0000 | 105.61 s | Achieves perfect Macro-F1 (1.000) but triples training time and increases gate noise vulnerability. |
| **2-Qubit VQC (1 Layer)** | 2 | 1 | 6 | 69.30% | 0.7960 | 0.7565 | 25.95 s | Information bottleneck from 2 PCA components limits classification. |
| **2-Qubit VQC (2 Layers)** | 2 | 2 | 12 | 69.30% | 0.9993 | 0.9993 | 35.51 s | Surprising expressivity; shows that 2 qubits can separate the data if given 2 entangling layers. |

