# QFF2026 QML Track — Results Report

**Team:** NEXUS-OPS  
**Selected problem statement:** PS-Q1 — Biomedical ECG Signal Classification  
**Report date:** October 30, 2026  
**Noise model:** QFF2026-v2

## 1. Approach Summary

NEXUS-OPS addresses PS-Q1 with a hybrid quantum–classical classification pipeline. The supplied dataset contains 30 statistical heartbeat features modelled on MIT-BIH statistics; these are feature vectors, not raw ECG recordings or direct patient recordings from MIT-BIH.

The VQC preprocessing pipeline standardizes the 30 input features, projects them to four principal components, and scales the components for angle encoding. The PCA model was fitted on the training partition. The four components retain **81.28%** of the training-set variance.

The classifier uses four qubits. Four input values are encoded with \(R_y\) rotations. The custom **NEXUS-Ring4** ansatz has two layers of trainable single-qubit Euler rotations (`qml.Rot`, equivalent to a Z–Y–Z rotation sequence) and bidirectional circular CNOT entanglers. It has **24 trainable parameters**. The measured expectation value of \(Z_0\) is mapped to an Arrhythmia score using:

\[
P(\mathrm{Arrhythmia}) = \frac{\langle Z_0\rangle+1}{2}.
\]

This is a custom ansatz; it is **not** PennyLane's built-in `StronglyEntanglingLayers` template.

## 2. PS-Q1 Classification Results

The official evaluation partition contains 1,500 labelled validation examples: 899 Normal (class 0) and 601 Arrhythmia (class 1). The reported metric is macro-averaged F1.

### Classical reference models

| Model | Validation accuracy | Validation macro-F1 | Verification status |
|---|---:|---:|---|
| Logistic Regression | 1.0000 | 1.0000 | Independently reproduced |
| Decision Tree (max depth 5) | 1.0000 | 1.0000 | Independently reproduced |

The supplied validation data are unusually easy to separate: a single-feature threshold using `w5_mean` also achieved 1.0000 validation accuracy and macro-F1 in the diagnostic audit. These results are important context when interpreting the VQC result.

Other baseline results and training times previously listed in this report are omitted because they have not been independently verified in this audit.

### Quantum classifier

| Evaluation | Accuracy | Macro-F1 | Confusion matrix |
|---|---:|---:|---|
| Ideal VQC | 1.0000 | 1.0000 | `[[899, 0], [0, 601]]` |
| Noisy VQC, QFF2026-v2, 8,192 shots/sample | 1.0000 | 1.0000 | `[[899, 0], [0, 601]]` |

The fresh noisy evaluation was rerun using the project's noisy-evaluation script and the organizer's QFF2026-v2 noise model. It evaluated all 1,500 validation examples. The generated JSON metrics and prediction CSV were independently checked: both contain 1,500 rows, no missing predictions, and metrics consistent with the confusion matrix above.

The noisy run used 8,192 shots per example, or 12,288,000 shots in total. The recorded run reported approximately 5.70 seconds for transpilation and 27.12 seconds for simulation. These are measurements from this run, not general hardware-performance estimates.

### Interpretation

The VQC retained perfect validation performance in this particular noisy simulation run. However, Logistic Regression and a shallow Decision Tree also achieved perfect validation scores, and the `w5_mean` feature alone separated the validation classes. Therefore, these results demonstrate that the VQC can solve the supplied benchmark under the tested simulator noise; **they do not demonstrate quantum advantage or superiority over classical machine learning**.

## 3. Task A — Bell State \(|\Phi^+\rangle\)

Task A was run using the organizer-provided `task_a_bell()` helper with 8,192 shots per measurement basis. The fidelity was calculated using the required correlation formula:

\[
F = \frac{1+\langle XX\rangle-\langle YY\rangle+\langle ZZ\rangle}{4}.
\]

The standard error was propagated from the three correlation standard errors.

| Condition | \(\langle XX\rangle\) | \(\langle YY\rangle\) | \(\langle ZZ\rangle\) | Fidelity |
|---|---:|---:|---:|---:|
| Ideal | +1.0000 ± 0.0000 | −1.0000 ± 0.0000 | +1.0000 ± 0.0000 | **1.0000 ± 0.0000** |
| Noisy, QFF2026-v2 | +0.8191 ± 0.0063 | −0.8279 ± 0.0062 | +0.8735 ± 0.0054 | **0.8801 ± 0.0026** |

Two independent direct executions of the official helper reproduced the saved Task A values. The noisy fidelity is lower than the ideal value, consistent with the effect of the organizer's gate and readout noise. The observed correlations quantify the degradation; this report does not attribute the difference between individual correlators to a specific noise mechanism without separate controlled experiments.

## 4. Task B — Processor D / QFT-3 Benchmark

The Processor D configuration was validated and the organizer's QFT-3 benchmark was independently rerun using the official command:

`python benchmark/qff_qml_benchmark.py configs/processor_d_config.json`

| Benchmark metric | Result |
|---|---:|
| Processor ID | `nexus_ops_ring4_v1` |
| Physical qubits | 4 |
| Coupling topology | Bidirectional ring |
| QFT-3 depth before transpilation | 6 |
| QFT-3 depth after transpilation | 21 |
| Two-qubit gate count | 12 |
| SWAP overhead | 1 |
| Ideal fidelity | 1.000000 |
| Noisy fidelity | 0.626784 |
| Benchmark noise model | QFF2026-v2 |

The official benchmark rerun reproduced the recorded values. The QFT-3 benchmark result is separate from the PS-Q1 classifier's noisy validation result; the two tasks use different circuits and evaluation procedures.

The ring coupling map also matches the circular entangling connectivity used by the custom NEXUS-Ring4 classifier circuit. For the classifier circuit, the verified QFF2026-v2 transpilation produced depth 15, eight CNOT gates, and zero SWAP gates. For QFT-3, the official benchmark reported one SWAP.

## 5. VQC Circuit and Resource Summary

| Resource | Verified value |
|---|---:|
| Qubits | 4 |
| Ansatz layers | 2 |
| Trainable parameters | 24 |
| Logical circuit depth (PennyLane resource specification) | 11 |
| QFF2026-v2 transpiled depth | 15 |
| Transpiled two-qubit gates | 8 CNOT gates |
| Transpiled SWAP gates | 0 |
| Noisy validation examples | 1,500 |
| Shots per noisy validation example | 8,192 |
| Total noisy validation shots | 12,288,000 |

The logical depth and transpiled depth refer to different circuit representations and should not be compared as if they were measured at the same compilation stage. The transpiled depth and gate counts above refer to the actual trained circuit evaluated by the noisy-evaluation implementation.

## 6. Dataset, Validation, and Limitations

1. **Dataset provenance.** The challenge documentation describes the supplied heartbeat feature vectors as modelled on MIT-BIH statistics. The supplied files do not establish that these are raw patient recordings or direct patient-level extracts from MIT-BIH.

2. **Strong class separability.** Logistic Regression, a shallow Decision Tree, and even a threshold on `w5_mean` achieved perfect validation performance in the independent diagnostic checks. The VQC's perfect score must be interpreted in this context. No claim of quantum advantage is supported.

3. **Patient-level independence cannot be confirmed from the supplied CSVs.** The supplied partitions contain no patient identifier, so patient-level separation cannot be independently audited from these files. Exact feature-row duplicate checks found no cross-partition duplicates; this does not prove patient-level independence or rule out near-duplicates.

4. **Internal holdout preprocessing caveat.** The VQC preprocessing objects were fitted on the training partition before the internal training/holdout split used for checkpoint selection. Consequently, the internal holdout was not completely isolated from preprocessing fitting. The official validation partition was transformed using the saved training-fitted preprocessing objects and was not used to fit those objects.

5. **Simulator rather than physical hardware.** The noisy VQC result was obtained with Qiskit Aer and the organizer's QFF2026-v2 noise model. It is not a result from a physical quantum processor and does not capture every hardware effect, such as calibration drift, crosstalk, or time-dependent noise.

6. **Single noisy evaluation run.** The reported 1.0000 noisy score is the result of the freshly executed 8,192-shot-per-example run. It should not be interpreted as a guarantee across other random seeds, noise realizations, or hardware devices.

7. **Notebook and automated tests.** The saved notebook was structurally inspected: its cells have execution counts and no saved error outputs. This inspection is not equivalent to a fresh end-to-end execution in a clean kernel. The pytest suite was not run in the active environment because `pytest` was not installed.

## 7. Reproducibility

The project uses fixed random seeds in the relevant scripts (including seed 42 where configured). The following commands are the principal project reproduction entry points; run them from the project root with the project environment activated:

```powershell
# Dataset audit
python src/data/dataset_audit.py

# Ideal VQC training/evaluation
python src/models/vqc_4qubit.py

# Task A — official Bell-state helper
python src/benchmarks/task_a_bell.py

# Task B — Processor D configuration and benchmark
python src/benchmarks/task_b_processor_d.py

# Task B — independently invoke the official benchmark
python "$env:QFF_KIT\benchmark\qff_qml_benchmark.py" configs\processor_d_config.json

# Noisy VQC validation
python -m src.models.vqc_noisy_evaluation

# Generate final test predictions
python src/models/generate_predictions.py
```

The `QFF_KIT` environment variable in the Task B command is optional convenience notation; replace it with the actual challenge-kit path if it is not defined. The official benchmark was independently invoked using the challenge-kit path in this audit.

## 8. Audit Notes

- Dataset audit: 10 of 10 checks passed.
- Task A: official helper executed independently twice; values reproduced.
- Task B: official benchmark independently rerun; values reproduced.
- Noisy VQC: fresh run completed; JSON and CSV metrics independently reconciled.
- Prediction CSV: 1,500 rows, exactly the required `sample_id,label` columns, no missing values or duplicate IDs, and IDs in the same order as the test feature file. This structural check does not independently establish the prediction-generation provenance.
- Automated pytest suite: not run because pytest was unavailable in the active environment.
- The original noisy-evaluation JSON and CSV were backed up before the fresh run. The backup is retained under `outputs/metrics/noisy_evaluation_backup_20261030_041024/`.

---

**Overall conclusion:** The project has reproduced the required Task A and Task B organizer benchmarks and achieved perfect ideal and noisy-simulator validation scores for PS-Q1 on the supplied dataset. The strongest scientifically defensible conclusion is benchmark feasibility and observed performance under the specified simulator noise. The supplied data's unusually strong classical separability means these results do not establish a quantum advantage.
