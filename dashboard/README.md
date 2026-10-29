# NEXUS-OPS Streamlit Dashboard

## Install (from the NEXUS_OPS_PS_Q1 project root)

```powershell
python -m pip install -r dashboard\requirements-dashboard.txt
```

## Run

```powershell
python -m streamlit run dashboard\app.py
```

Open the local URL Streamlit prints (normally http://localhost:8501).

The app reads `outputs/metrics/*.json`, `outputs/predictions/predictions_q1.csv`,
and `reports/results.md`. It does not train models or modify submission artifacts.

**Important:** It displays validation metrics. Because test labels are hidden, the
test prediction CSV can only be checked for structure and class distribution; it
cannot provide test accuracy.
