# NEXUS-OPS | Qiskit Fall Fest 2026 — Phase 2 Deliverable

**Team Name:** NEXUS-OPS  
**Competition:** Qiskit Fall Fest 2026 (Phase 2 Hackathon)  
**Selected Problem Statement:** **PS-Q1 — Biomedical Signal Classification (ECG Arrhythmia Detection)**  
**Common Tasks Included:** **Task A (Bell-State Preparation & Fidelity)** and **Task B (Processor D Design & QFT-3 Benchmark)**  
**Track:** Quantum Machine Learning (QML)  
**Submission Portal:** [https://qffsrmap2026.com/learning/hackathon](https://qffsrmap2026.com/learning/hackathon)  
**Noise Model:** Organizer Benchmark `QFF2026-v2`  
**Fixed Random Seed:** `42`  

---

## 1. Executive Summary & Key Results

NEXUS-OPS delivers an end-to-end, scientifically grounded, fully reproducible Quantum Machine Learning solution for classifying 30-feature electrocardiogram (ECG) heartbeats into Normal (`0`) or Arrhythmia (`1`).

| Metric / Benchmark | Classical Baseline (Logistic Regression) | 4-Qubit VQC (Ideal Simulation) | 4-Qubit VQC (Noisy `QFF2026-v2`, 8192 shots) |
| :--- | :--- | :--- | :--- |
| **Validation Macro-F1** | **1.0000** | **0.9979** | **0.9965** |
| **Validation Accuracy** | **1.0000** | **0.9980** | **0.9967** |
| **Validation Precision (Macro)** | 1.0000 | 0.9983 | 0.9972 |
| **Validation Recall (Macro)** | 1.0000 | 0.9975 | 0.9958 |
| **Confusion Matrix (Normal / Arrhythmia)** | [899, 0] / [0, 601] | [899, 0] / [3, 598] | [899, 0] / [5, 596] |
| **Trainable Parameters** | 31 | 24 | 24 |
| **Circuit Depth** | N/A | 8 (raw) / 13 (transpiled) | 13 (transpiled to `rx, ry, rz, cx`) |
| **Training Time** | 1.155 s | 40.08 s (10 epochs, 6,000 samples) | Pretrained weights evaluated under noise |

### Common Tasks Official Results:
- **Task A (Bell State $|\Phi^+\rangle$):**
  - **Ideal Fidelity:** $1.0000 \pm 0.0000$
  - **Noisy Fidelity (`QFF2026-v2`):** $0.8801 \pm 0.0026$ ($\langle XX \rangle = +0.8191$, $\langle YY \rangle = -0.8279$, $\langle ZZ \rangle = +0.8735$)
- **Task B (Processor D — `nexus_ops_ring4_v1`):**
  - **Topology:** 4-qubit bidirectional cyclic ring (matches 4-qubit cyclic ansatz with **0 SWAP overhead**)
  - **QFT-3 Benchmark Ideal Fidelity:** **1.000000**
  - **QFT-3 Benchmark Noisy Fidelity (`QFF2026-v2`):** **0.626784**
  - **QFT-3 Circuit Depth:** 6 (before transpile) $\rightarrow$ 21 (after transpile)
  - **QFT-3 Two-Qubit Gates:** 12 CX gates (SWAP overhead: 1)
- **Final Test Predictions:**
  - File: `outputs/predictions/predictions_q1.csv` (1,500 rows, `s0` to `s1499`)
  - Status: **PASSED official `tools/validate_predictions.py q1`**

---

## 2. Repository Structure

```text
NEXUS_OPS_PS_Q1/
├── configs/
│   └── processor_d_config.json          # Task B Processor D configuration (validated)
├── notebooks/
│   └── solution.ipynb                   # Executed top-to-bottom master Jupyter notebook
├── outputs/
│   ├── figures/
│   │   ├── ablation_study_chart.png     # Performance vs compute cost ablation chart
│   │   ├── confusion_matrix_comparison.png # Side-by-side Ideal vs Noisy confusion matrices
│   │   └── vqc_training_loss.png        # 10-epoch training MSE loss trajectory
│   ├── metrics/
│   │   ├── classical_baseline.json      # Classical Logistic Regression validation metrics
│   │   ├── vqc_4qubit_ideal.json        # Ideal 4-qubit VQC validation metrics
│   │   ├── vqc_4qubit_noisy.json        # Noisy 4-qubit VQC validation metrics
│   │   ├── task_a_bell.json             # Official Task A Bell state fidelity metrics
│   │   └── ablation_study.json          # Complete 5-model ablation study data
│   ├── models/
│   │   ├── logistic_regression_baseline.joblib # Trained baseline model
│   │   ├── vqc_4qubit_weights.npy       # Trained 24-parameter VQC weights
│   │   └── vqc_4qubit_preprocessing.joblib    # Preprocessors fitted strictly on train
│   └── predictions/
│       └── predictions_q1.csv           # Official 1,500 test predictions (validated)
├── reports/
│   ├── dataset_audit.json               # Automated dataset schema & integrity audit
│   └── results.md                       # Official Results Report (filled from template)
├── src/
│   ├── benchmarks/
│   │   ├── classical_baseline.py        # Classical benchmark execution script
│   │   ├── task_a_bell.py               # Official Task A runner
│   │   └── task_b_processor_d.py        # Official Task B runner and validator
│   ├── data/
│   │   ├── dataset_audit.py             # Dataset validation script
│   │   └── generate_figures.py          # Figure rendering script
│   └── models/
│       ├── vqc_4qubit.py                # Vectorized 4-qubit VQC training script
│       ├── vqc_noisy_evaluation.py      # QFF2026-v2 noisy evaluation script
│       ├── ablation_study.py            # Layer and qubit ablation experiment runner
│       └── generate_predictions.py      # Final test prediction & validation script
├── tests/
│   ├── run_all_tests.py                 # Master zero-dependency test runner
│   ├── test_dataset.py                  # Dataset schema, null, and dimension tests
│   ├── test_leakage_and_preprocessing.py# Data leakage prevention verification
│   ├── test_models.py                   # Model weights, dimensions, and metric thresholds
│   ├── test_task_a.py                   # Task A Bell state fidelity tests
│   ├── test_task_b.py                   # Task B Processor D schema & QFT-3 tests
│   └── test_predictions.py             # Official predictions file validation
├── solution.ipynb                       # Root copy of executed master notebook
├── requirements.txt                     # Pinned environment packages
├── requirements-lock.txt                # Full pip freeze environment lock
└── README.md                            # Complete documentation & reproduction guide
```

---

## 3. Environment Setup & Reproduction

### Prerequisites
- Python 3.11, 3.12, or 3.13 (Verified on Python 3.13.7, Windows 11 / Linux / macOS)
- Dedicated virtual environment `qml-env`

### Installation
```bash
# 1. Clone the repository
git clone <YOUR_REPO_URL>
cd NEXUS_OPS_PS_Q1

# 2. Create and activate a dedicated virtual environment
python -m venv qml-env
# Windows PowerShell:
.\qml-env\Scripts\Activate.ps1
# Linux / macOS:
source qml-env/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

### Reproducing All Results (Step-by-Step)

```bash
# Step 1: Run Dataset Audit
python src/data/dataset_audit.py

# Step 2: Run Classical Baseline
python src/benchmarks/classical_baseline.py

# Step 3: Train 4-Qubit VQC (Ideal Simulation)
python src/models/vqc_4qubit.py

# Step 4: Run Task A (Bell-State Fidelity Benchmark)
python src/benchmarks/task_a_bell.py

# Step 5: Run Task B (Processor D Benchmark & Validation)
python src/benchmarks/task_b_processor_d.py

# Step 6: Run Noisy VQC Evaluation under QFF2026-v2
python src/models/vqc_noisy_evaluation.py

# Step 7: Run Ablation Studies (1 vs 2 vs 3 Layers, 2 vs 4 Qubits)
python src/models/ablation_study.py

# Step 8: Generate Final Test Predictions & Run Official Validator
python src/models/generate_predictions.py

# Step 9: Execute Automated Test Suite (6/6 Passed)
python tests/run_all_tests.py
```

### Running the Notebook Top-to-Bottom
You can run the master notebook in Jupyter or execute it headlessly via `nbconvert`:
```bash
jupyter nbconvert --to notebook --execute solution.ipynb --output solution.ipynb
```

---

## 4. Judging Rubric Compliance Matrix

| Criterion | Rubric Weight | NEXUS-OPS Implementation & Evidence |
| :--- | :--- | :--- |
| **G1 Deadline** | Gate | Submitted to portal before deadline. |
| **G2 Scope** | Gate | PS-Q1 + Task A + Task B documented in README. |
| **G3 Predictions** | Gate | `outputs/predictions/predictions_q1.csv` passes `tools/validate_predictions.py q1`. |
| **G4 Notebook** | Gate | `solution.ipynb` executes end-to-end without errors. |
| **G5 Leakage** | Gate | All scaling and PCA parameters fitted strictly on `ecg_train.csv`. Verified by `test_leakage_and_preprocessing.py`. |
| **C1 Quantum Design** | 30 pts | 4-qubit register, $R_y$ angle encoding, 2 StronglyEntanglingLayers (24 parameters). Empirical resource accounting and PCA justification. |
| **C2 Results Quality** | 25 pts | Honest classical vs quantum comparison. Classical Macro-F1 = 1.0000; VQC Macro-F1 = 0.9979. Consistent validation metrics. |
| **C3 Noise Analysis** | 20 pts | Task A fidelity ($0.8801 \pm 0.0026$). Noisy VQC under `QFF2026-v2` ($0.9965$ Macro-F1, transpiled to `rx, ry, rz, cx`, 8192 shots). |
| **C4 Processor D** | 15 pts | `nexus_ops_ring4_v1` bidirectional ring. Passes `qff_qml_benchmark.py`. Zero SWAP overhead for cyclic ansatz. QFT-3 noisy fidelity = 0.6268. |
| **C5 Reproducibility** | 10 pts | Pinned `requirements-lock.txt`, `SEED=42`, complete `results.md`, automated test suite. |
| **Bonus Points** | +5 pts | Extensive 5-configuration ablation study (depth, dimension, parameters) and clinical problem framing. |

---

## 5. Pinned Dependencies (`pip freeze`)

```text
annotated-types==0.8.0
anyio==4.15.1
appdirs==1.4.4
argon2-cffi==25.1.0
argon2-cffi-bindings==26.1.0
arrow==1.4.0
asttokens==3.0.2
astunparse==1.6.3
async-lru==2.4.0
attrs==26.1.0
autograd==1.9.1
autoray==0.8.2
babel==2.18.0
beautifulsoup4==4.15.0
bleach==6.4.0
cachetools==7.2.7
certifi==2026.10.23
cffi==2.1.1
charset-normalizer==3.5.2
cloudpickle==3.1.2
colorama==0.4.6
comm==0.2.3
contourpy==1.4.0
cryptography==50.0.2
cycler==0.12.1
debugpy==1.8.22
defusedxml==0.7.1
diastatic-malt==2.15.2
dill==0.4.1
executing==2.3.0
fastjsonschema==2.22.2
fonttools==4.68.0
fqdn==1.6.0
gast==0.7.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
ibm-cloud-sdk-core==3.28.0
ibm-platform-services==0.77.1
idna==3.20
ipykernel==7.4.0
ipython==9.17.1
ipython_pygments_lexers==1.1.1
isoduration==20.11.0
jedi==0.20.1
Jinja2==3.1.6
joblib==1.6.0
json5==0.17.3
jsonpointer==3.6
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter_builder==1.2.4
jupyter_client==8.10.0
jupyter_core==5.9.1
jupyter-events==0.12.1
jupyter-lsp==2.3.1
jupyter_server==2.21.1
jupyter_server_terminals==0.5.4
jupyterlab==4.6.4
jupyterlab_pygments==0.3.0
jupyterlab_server==2.28.1
kiwisolver==1.5.1
lark==1.3.1
lxml==6.1.3
MarkupSafe==3.0.4
matplotlib==3.11.2
matplotlib-inline==0.2.2
mistune==3.4.0
mpmath==1.3.0
narwhals==2.28.0
nbclient==0.11.0
nbconvert==7.17.2
nbformat==5.11.1
nest-asyncio2==1.7.4
networkx==3.7
notebook==7.6.3
notebook_shim==0.2.4
numpy==2.4.6
packaging==26.3
pandas==3.1.0
pandocfilters==1.5.1
parso==0.8.7
pennylane==0.44.1
pennylane_lightning==0.45.0
PennyLane-qiskit==0.44.1
pillow==13.0.0
pip==26.2.2
platformdirs==4.14.0
prometheus_client==0.26.0
prompt_toolkit==3.0.53
psutil==7.2.2
pure_eval==0.2.4
pycparser==3.12
pydantic==2.14.1
pydantic_core==2.50.1
Pygments==2.21.0
PyJWT==2.15.1
pyparsing==3.3.3
pypdf==6.21.4
pyspnego==0.12.4
python-dateutil==2.9.0.post0
python-docx==1.2.0
python-json-logger==4.2.0
pywinpty==3.1.2
PyYAML==6.0.3
pyzmq==27.2.0
qiskit==2.3.0
qiskit-aer==0.17.2
qiskit-ibm-runtime==0.45.1
referencing==0.37.0
requests==2.34.2
requests_ntlm==1.3.0
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rfc3987-syntax==1.1.0
rpds-py==2026.9.1
rustworkx==0.18.1
scikit-learn==1.9.1
scipy==1.17.1
scipy-openblas32==0.3.34.237.2
Send2Trash==2.1.0
six==1.17.0
soupsieve==3.0.4
sspilib==0.6.0
stack-data==0.6.3
stevedore==5.9.1
sympy==1.14.0
termcolor==3.3.0
terminado==0.18.1
threadpoolctl==3.7.0
tinycss2==1.5.1
tomlkit==0.15.2
tornado==6.6
traitlets==5.16.1
typing_extensions==4.16.0
typing-inspection==0.4.4
tzdata==2026.6
uri-template==1.3.0
urllib3==2.8.0
wcwidth==0.9.4
webcolors==25.10.0
webencodings==0.6.1
websocket-client==1.9.2
wheel==0.48.0
```

