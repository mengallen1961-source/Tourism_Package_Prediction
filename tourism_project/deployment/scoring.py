# ==============================================================
# Cell 8.1 | Write the scoring module used by the app
# --------------------------------------------------------------
# Purpose : Keep all model logic (load, validate, score, rank) outside the
#           user interface so that it can be tested without a browser.
# Input   : None (functions read the bundle at run time)
# Output  : tourism_project/deployment/scoring.py
# ==============================================================
"""Model logic for the Streamlit app: load the bundle, validate inputs, score customers."""
import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_FILE, METRICS_FILE = "tourism_model.joblib", "metrics.json"


def load_bundle(model_dir):
    """Return the model bundle (dict) committed by the pipeline, or None if it does not exist yet."""
    path = Path(model_dir) / MODEL_FILE
    return joblib.load(path) if path.exists() else None


def load_report_card(model_dir):
    path = Path(model_dir) / METRICS_FILE
    return json.loads(path.read_text()) if path.exists() else None


def validate(bundle, df):
    """Check a dataframe against the bundle's input schema. Returns a list of problems (empty = valid)."""
    problems = []
    missing = [f for f in bundle["features"] if f not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    for feature, spec in bundle["input_schema"].items():
        col = df[feature]
        if col.isna().any():
            problems.append(f"{feature}: {int(col.isna().sum())} empty value(s)")
        elif spec["type"] == "categorical":
            unknown = sorted(set(col.astype(str)) - set(spec["levels"]))
            if unknown:
                problems.append(f"{feature}: unknown value(s) {unknown}; expected one of {spec['levels']}")
        else:
            numeric = pd.to_numeric(col, errors="coerce")
            if numeric.isna().any():
                problems.append(f"{feature}: non-numeric value(s)")
    return problems


def out_of_range(bundle, df):
    """Numeric values outside the range seen in training: the model has no experience there."""
    notes = []
    for feature, spec in bundle["input_schema"].items():
        if spec["type"] == "numeric" and feature in df.columns:
            values = pd.to_numeric(df[feature], errors="coerce")
            n = int(((values < spec["min"]) | (values > spec["max"])).sum())
            if n:
                notes.append(f"{feature}: {n} value(s) outside the training range "
                             f"{spec['min']:,.0f}-{spec['max']:,.0f}; treat the score with caution")
    return notes


def score(bundle, df):
    """Purchase probability and decision for every row (columns in the bundle's feature order)."""
    X = df[bundle["features"]].copy()
    for feature, spec in bundle["input_schema"].items():
        if spec["type"] == "numeric":
            X[feature] = pd.to_numeric(X[feature])
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    out = df.copy()
    out["purchase_probability"] = proba.round(4)
    out["recommendation"] = ["call" if p >= bundle["threshold"] else "do not call" for p in proba]
    return out


def rank_call_list(bundle, df):
    """Score a batch and sort it from the most to the least likely buyer."""
    scored = score(bundle, df).sort_values("purchase_probability", ascending=False).reset_index(drop=True)
    scored.insert(0, "rank", range(1, len(scored) + 1))
    return scored
