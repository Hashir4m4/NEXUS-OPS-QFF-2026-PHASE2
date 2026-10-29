import json
from pathlib import Path
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "configs" / "processor_d_config.json") as f:
    cfg = json.load(f)

cmap = CouplingMap(cfg["coupling_map"])
basis = cfg["native_gate_set"]

# Build VQC circuit
qc = QuantumCircuit(4, 1)
for i in range(4):
    qc.ry(0.5, i)
for l in range(2):
    for q in range(4):
        qc.rz(0.1, q); qc.ry(0.2, q); qc.rz(0.3, q)
    if l == 0:
        qc.cx(0, 1); qc.cx(1, 2); qc.cx(2, 3); qc.cx(3, 0)
    else:
        qc.cx(0, 2); qc.cx(1, 3); qc.cx(2, 0); qc.cx(3, 1)
qc.measure(0, 0)

print("Original VQC ops:", dict(qc.count_ops()))
print("Original VQC depth:", qc.depth())

# Transpile to Processor D
for opt in [1, 2, 3]:
    tqc = transpile(qc, coupling_map=cmap, basis_gates=basis, optimization_level=opt, seed_transpiler=42)
    ops = dict(tqc.count_ops())
    print(f"Processor D (opt level {opt}): depth={tqc.depth()}, ops={ops}")
