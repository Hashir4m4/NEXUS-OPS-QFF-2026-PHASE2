from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

# Run this app from the NEXUS_OPS_PS_Q1 project root.
ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "outputs" / "metrics"
PREDICTIONS = ROOT / "outputs" / "predictions" / "predictions_q1.csv"
RESULTS = ROOT / "reports" / "results.md"

st.set_page_config(page_title="NEXUS-OPS | ECG Performance", page_icon="🫀", layout="wide")
st.title("🫀 NEXUS-OPS | ECG Model Performance")
st.caption("Qiskit Fall Fest 2026 · QML · PS-Q1: Normal vs Arrhythmia")

with st.sidebar:
    st.header("Controls")
    st.caption("Reads saved project artifacts; does not retrain models.")
    if st.button("Refresh dashboard"):
        st.cache_data.clear()
        st.rerun()
    st.info("Macro-F1 is the primary comparison metric. Test labels are hidden.")

def read_json(filename):
    path = METRICS_DIR / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"Could not read {filename}: {exc}")
        return None

def find_metric(obj, names):
    if isinstance(obj, dict):
        for name in names:
            value = obj.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        for value in obj.values():
            found = find_metric(value, names)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_metric(value, names)
            if found is not None:
                return found
    return None

def metric(data, keys, fallback):
    value = find_metric(data, keys) if data else None
    if value is None:
        return fallback, "completion-report fallback"
    if 1 < value <= 100:
        value /= 100
    return value, "metrics JSON"

# Fallback values are taken from the team's supplied completion report.
specs = {
    "Classical baseline": ("classical_baseline.json", ("macro_f1", "f1_macro", "macro_f1_score"), ("accuracy", "val_accuracy"), 1.0000, 1.0000),
    "Ideal VQC": ("vqc_4qubit_ideal.json", ("macro_f1", "f1_macro", "macro_f1_score"), ("accuracy", "val_accuracy"), 0.9979, 0.9980),
    "Noisy VQC": ("vqc_4qubit_noisy.json", ("macro_f1", "f1_macro", "macro_f1_score"), ("accuracy", "val_accuracy"), 0.9965, 0.9967),
}
rows = []
for name, (file, f1keys, acckeys, f1fallback, accfallback) in specs.items():
    data = read_json(file)
    f1, f1source = metric(data, f1keys, f1fallback)
    acc, accsource = metric(data, acckeys, accfallback)
    rows.append({"Model": name, "Macro-F1": f1, "Accuracy": acc, "Source": f1source if f1source == accsource else f"{f1source}; {accsource}"})
scores = pd.DataFrame(rows)

st.subheader("Validation performance")
a, b, c, d = st.columns(4)
a.metric("Classical Macro-F1", f"{scores.iloc[0]['Macro-F1']:.4f}")
b.metric("Ideal VQC Macro-F1", f"{scores.iloc[1]['Macro-F1']:.4f}")
c.metric("Noisy VQC Macro-F1", f"{scores.iloc[2]['Macro-F1']:.4f}")
d.metric("Noisy VQC accuracy", f"{scores.iloc[2]['Accuracy']:.2%}")
st.warning("These are validation metrics, not hidden-test accuracy. Test labels are unavailable.")

left, right = st.columns(2)
with left:
    fig = px.bar(scores, x="Model", y="Macro-F1", color="Model", text_auto=".4f", range_y=[0, 1.05], title="Macro-F1 comparison")
    fig.update_layout(showlegend=False, yaxis_title="Macro-F1")
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.bar(scores, x="Model", y="Accuracy", color="Model", text_auto=".4f", range_y=[0, 1.05], title="Validation accuracy")
    fig.update_layout(showlegend=False, yaxis_title="Accuracy")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Metric table")
st.dataframe(scores.style.format({"Macro-F1": "{:.4f}", "Accuracy": "{:.4f}"}), use_container_width=True, hide_index=True)

st.subheader("Validation confusion matrices")
st.caption("Counts below are from the supplied completion report.")
x, y = st.columns(2)
with x:
    st.plotly_chart(px.imshow([[899, 0], [3, 598]], text_auto=True, aspect="auto",
        x=["Predicted Normal", "Predicted Arrhythmia"], y=["Actual Normal", "Actual Arrhythmia"],
        title="Ideal VQC · 1,500 validation samples", color_continuous_scale="Blues"), use_container_width=True)
with y:
    st.plotly_chart(px.imshow([[899, 0], [5, 596]], text_auto=True, aspect="auto",
        x=["Predicted Normal", "Predicted Arrhythmia"], y=["Actual Normal", "Actual Arrhythmia"],
        title="Noisy VQC · 1,500 validation samples", color_continuous_scale="Blues"), use_container_width=True)

st.subheader("Noise impact")
drop = scores.iloc[1]["Macro-F1"] - scores.iloc[2]["Macro-F1"]
n1, n2, n3 = st.columns(3)
n1.metric("Ideal VQC Macro-F1", f"{scores.iloc[1]['Macro-F1']:.4f}")
n2.metric("Noisy VQC Macro-F1", f"{scores.iloc[2]['Macro-F1']:.4f}")
n3.metric("Macro-F1 decrease", f"{drop:.4f}")

st.subheader("Test prediction file")
if PREDICTIONS.exists():
    try:
        pred = pd.read_csv(PREDICTIONS)
        if not {"sample_id", "label"}.issubset(pred.columns):
            st.error("CSV must contain sample_id,label columns.")
        else:
            p, q, r = st.columns(3)
            p.metric("Prediction rows", f"{len(pred):,}")
            q.metric("Unique sample IDs", f"{pred.sample_id.nunique():,}")
            r.metric("Labels present", str(sorted(pred.label.dropna().unique().tolist())))
            ok = len(pred) == 1500 and pred.sample_id.nunique() == 1500 and set(pred.label.dropna().unique()).issubset({0, 1})
            st.success("Basic CSV structure checks passed.") if ok else st.warning("One or more basic CSV checks failed.")
            dist = pred.label.value_counts().reindex([0, 1], fill_value=0).rename(index={0: "Normal (0)", 1: "Arrhythmia (1)"}).rename_axis("Predicted class").reset_index(name="Count")
            l, r = st.columns(2)
            l.plotly_chart(px.pie(dist, names="Predicted class", values="Count", hole=.4, title="Test prediction distribution"), use_container_width=True)
            r.dataframe(pred.head(20), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Unable to read prediction CSV: {exc}")
else:
    st.warning(f"Prediction file not found: {PREDICTIONS}")

st.subheader("Task A, Task B and ablation artifacts")
artifact_names = ["task_a_bell.json", "task_b_processor_d.json", "ablation_study.json", "dataset_audit.json"]
artifact_status = pd.DataFrame([{"Artifact": f, "Status": "Found" if (METRICS_DIR / f).exists() else "Not found"} for f in artifact_names])
st.dataframe(artifact_status, use_container_width=True, hide_index=True)
with st.expander("Reported benchmark reference values"):
    st.write("Task A Bell fidelity: ideal 1.0000 ± 0.0000; noisy 0.8801 ± 0.0026.")
    st.write("Task B Processor D QFT-3 fidelity: ideal 1.000000; noisy 0.626784.")
    st.caption("These reference values are from the supplied project completion report.")

if RESULTS.exists():
    with st.expander("View reports/results.md"):
        st.markdown(RESULTS.read_text(encoding="utf-8"))

st.divider()
st.caption("Competition/research dashboard only. This model is not a clinically validated diagnostic system.")
