# NEXUS-OPS Phase 2 — Repair & Rubric Compliance Plan

**Objective:** Maximize legitimate score on the official `QFF_2026_Phase2_QML_Judging_Rubric_2page.pdf` (Gates G1–G5, C1 Quantum Design /30, C2 Results /25, C3 Noise /20, C4 Processor D /15, C5 Reproducibility /10, Bonus /5, capped at 100).  
**Strategy:** Prioritize eligibility gates > correctness/consistency > reproducibility > extras. Commit after each completed item.

---

## 1. Prioritized Work Items & Execution Sequence

### Item 1: Make Results Consistent (Highest Impact)
- **Defect:** Discrepancy between `solution.ipynb` outputs (Macro-F1 ~0.9944 / ~0.9951, confusion [[899,0],[8,593]] / [[899,0],[7,594]]) and README / `results.md` (0.9979 / 0.9965, 3 and 5 FN).
- **Action:**
  1. Audit `outputs/models/vqc_4qubit_weights.npy` and `src/models/vqc_4qubit.py` to identify which weights generated `predictions_q1.csv` and `outputs/metrics/vqc_4qubit_noisy.json`.
  2. Harmonize the training/evaluation code so that running the pipeline reproduces identical results in both standalone scripts and `solution.ipynb`.
  3. Ensure `solution.ipynb` and `notebooks/solution.ipynb` are synchronized or consolidated via `tools/build_notebook.py`.
  4. Ensure all metrics JSONs, CSVs, `solution.ipynb`, `README.md`, and `reports/results.md` display the exact same observed numbers.

### Item 2: Remove or Fix Unsupported Claims
- **Defect A (Zero-SWAP Claim):** The second entangling layer uses CX(0,2), CX(1,3), CX(2,0), CX(3,1), which are cross-ring (distance 2) on a 4-ring. 
  - *Action:* Honestly rewrite the claim in `README.md` and `reports/results.md`. State clearly that the VQC's second entangling layer requires non-adjacent connectivity; transpile it to Processor D's ring coupling map and report the exact SWAP/depth routing overhead.
- **Defect B (Mitigation Row 0.9972):**
  - *Action:* Check repository for actual implementation code. If absent or unreproducible, remove the row from `reports/results.md` and state honestly that error mitigation was not applied to PS-Q1.
- **Defect C (Classical Baselines):**
  - *Action:* Verify whether Linear SVM, RBF SVM, and Random Forest have reproducible scripts and JSON outputs. If only Logistic Regression has a saved artifact, run a script to generate the baselines with train-only scaling or clearly label unverified baselines.
- **Defect D (Overclaims):**
  - *Action:* Soften assertions such as "exceptional inductive bias" and "fully reproducible" until cleanly verified.

### Item 3: Eligibility Gates (G1–G5)
- **G1:** Submission repository URL submitted before deadline (record external verification via portal).
- **G2:** Explicitly verify that README, notebook, and all reports name **PS-Q1 + Task A + Task B**.
- **G3:** Re-run `tools/validate_predictions.py q1 outputs/predictions/predictions_q1.csv` and log raw stdout.
- **G4:** Execute `solution.ipynb` end-to-end from repo root in `qml-env` with zero errors.
- **G5:** Audit data leakage across all steps (StandardScaler, PCA, MinMaxScaler, angle encoding, checkpoint selection, predictions). Add regression test asserting that passing test data into fit raises an error or fails.

### Item 4: Noise Analysis (C3)
- **Task A:** Rerun `src/benchmarks/task_a_bell.py` (8192 shots, XX/YY/ZZ parities, $F = (1+\langle XX \rangle - \langle YY \rangle + \langle ZZ \rangle)/4$). Record empirical noisy fidelity and standard error.
- **VQC Noisy Run:** Transpile to basis `['rx', 'ry', 'rz', 'cx']` with `QFF2026-v2` noise model and 8192 shots.
- **Q1 Noise Dynamics Analysis:** Quantify and explain the distribution of $\langle Z_0 \rangle$ under ideal vs noisy simulation, explaining why readout and gate depolarizing shift expectations toward 0 (the 0.5 decision boundary) and cause subtle FN increases.

### Item 5: Task B — Processor D (C4)
- Validate `configs/processor_d_config.json` with official `benchmark/qff_qml_benchmark.py`.
- Report transpiled depth, 2-qubit gates, SWAP overhead, and fidelity.
- Update architecture rationale to accurately discuss the ring topology trade-off (degree-2 minimizes crosstalk/frequency crowding, but introduces 1 SWAP for QFT-3 and routing overhead for range-2 VQC layers).

### Item 6: Quantum Design & Controlled Ablations (C1, C2)
- Document design justification: PCA dimensionality reduction (81.3% variance) to match 4 qubits, $R_y$ angle embedding, and variational layers.
- Verify ablation study with isolated seeds across layer depths and qubit counts (2 vs 4 qubits).

### Item 7: Centralized Paths & Clean Tests (C5)
- Verify `src/utils/config.py` is utilized across all modules.
- Ensure automated test suite runs zero-dependency in `qml-env`.

### Item 8: Reporting & Rubric Compliance Document
- Update `reports/results.md`, `README.md`, `reports/verification_log.md`.
- Create `reports/rubric_compliance.md` evaluating all Gates G1-G5 and Categories C1-C5 with point allocations and risk assessments.
