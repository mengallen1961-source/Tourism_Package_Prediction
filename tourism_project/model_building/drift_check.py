# ==============================================================
# Cell 6.1 | Write the drift check script
# --------------------------------------------------------------
# Purpose : Create drift_check.py, the third job of the pipeline. It also
#           provides the profile/fingerprint helpers that train.py reuses.
# Input   : None (the script reads its inputs at run time, see below)
# Output  : tourism_project/model_building/drift_check.py
# ==============================================================
"""
Decide whether the model needs retraining.

Input : artifacts/data/Xtrain.csv, ytrain.csv          (current prepared training data)
        <deploy-dir>/reference_profile.json           (profile of the live model's training data)
        pipeline code and configuration               (fingerprinted)
Output: artifacts/drift_report.json; job output `retrain=true|false` (GitHub Actions);
        step summary on the workflow run page
Exit  : 0 = decision made (retrain or skip), 1 = error
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("artifacts") / "data"
DEFAULT_DEPLOY_DIR = PROJECT_ROOT / "deployment"
DEFAULT_REPORT = Path("artifacts") / "drift_report.json"
PROFILE_FILE = "reference_profile.json"

# Policy
PSI_ALERT = 0.20                 # significant shift -> retrain
PSI_WATCH = 0.10                 # moderate shift -> reported only
MIN_NEW_DATA_SHARE = 0.10        # retrain when the training set grew by >= 10 %
MAX_LEVELS_AS_CATEGORY = 10      # numeric features with few values are profiled level by level
CODE_FILES = ["model_building/data_register.py", "model_building/prep.py",
              "model_building/train.py", "config/model_config.json"]


# ---------------------------------------------------------------- shared helpers (also used by train.py)
def file_sha256(*paths):
    """Fingerprint of one or more files."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def code_fingerprint(project_root=PROJECT_ROOT):
    """Fingerprint of everything that defines how the model is built."""
    digest = hashlib.sha256()
    for rel in CODE_FILES:
        path = Path(project_root) / rel
        digest.update(rel.encode())
        digest.update(path.read_bytes() if path.exists() else b"<missing>")
    return digest.hexdigest()


def build_profile(X, y):
    """Per-feature bins and shares of a training set (the reference for later drift checks)."""
    features = {}
    for col in X.columns:
        values = X[col]
        if values.dtype == object or values.nunique() <= MAX_LEVELS_AS_CATEGORY:
            shares = values.astype(str).value_counts(normalize=True).sort_index()
            features[col] = {"kind": "levels", "shares": shares.round(6).to_dict()}
        else:
            inner = np.unique(np.quantile(values.dropna(), np.linspace(0, 1, 11)))[1:-1]
            counts = np.histogram(values.dropna(), bins=np.r_[-np.inf, inner, np.inf])[0]
            features[col] = {"kind": "bins", "inner_edges": inner.round(6).tolist(),
                             "shares": (counts / counts.sum()).round(6).tolist()}
    return {"n_rows": int(len(X)), "target_rate": float(y.mean()), "features": features}


def psi(reference, current, eps=1e-4):
    """Population Stability Index between two share vectors (zero shares floored at eps)."""
    ref = np.clip(np.asarray(reference, dtype=float), eps, None)
    cur = np.clip(np.asarray(current, dtype=float), eps, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def feature_psi(spec, values):
    """PSI of one feature: current values binned exactly like the reference."""
    if spec["kind"] == "levels":
        cur = values.astype(str).value_counts(normalize=True)
        levels = sorted(set(spec["shares"]) | set(cur.index))          # unseen levels count too
        return psi([spec["shares"].get(l, 0.0) for l in levels], [cur.get(l, 0.0) for l in levels])
    counts = np.histogram(values.dropna(), bins=np.r_[-np.inf, spec["inner_edges"], np.inf])[0]
    return psi(spec["shares"], counts / counts.sum())


# ---------------------------------------------------------------- decision
def decide(reference, X, y, data_sha, code_sha, force=False):
    """Return (retrain, reasons, notes, psi_table, growth). Reasons trigger retraining; notes are warnings."""
    reasons, notes, psi_table = [], [], {}
    if force:
        reasons.append("forced by the operator (workflow_dispatch input)")
    if reference is None:
        reasons.append("no live model yet")
        return True, reasons, notes, psi_table, None
    if code_sha != reference["code_sha256"]:
        reasons.append("code or configuration changed since the live model was trained")
    growth = (len(X) - reference["n_rows"]) / reference["n_rows"]
    if growth >= MIN_NEW_DATA_SHARE:
        reasons.append(f"enough new data: {len(X):,} training customers vs {reference['n_rows']:,} (+{growth:.1%})")
    for col in reference["model_features"]:
        if col in X.columns:
            psi_table[col] = round(feature_psi(reference["features"][col], X[col]), 4)
    drifted = {c: v for c, v in psi_table.items() if v >= PSI_ALERT}
    if drifted:
        reasons.append(f"data drift (PSI >= {PSI_ALERT}): " + ", ".join(f"{c} {v:.3f}" for c, v in drifted.items()))
    # Warnings that do not trigger retraining on their own
    watch = {c: v for c, v in psi_table.items() if PSI_WATCH <= v < PSI_ALERT}
    if watch:
        notes.append("watch (PSI 0.10-0.20): " + ", ".join(f"{c} {v:.3f}" for c, v in watch.items()))
    if growth <= -MIN_NEW_DATA_SHARE:
        notes.append(f"training data shrank by {-growth:.1%}: check the source data")
    if not reasons and data_sha != reference["train_data_sha256"]:
        notes.append("data changed, but below every retraining threshold")
    return bool(reasons), reasons, notes, psi_table, growth


def write_outputs(report):
    """GitHub Actions: job output for the workflow + summary on the run page."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as fh:
            fh.write(f"retrain={'true' if report['retrain'] else 'false'}\n")
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        lines = ["## Drift check", f"**Retrain:** {'yes' if report['retrain'] else 'no'}  "]
        lines += [f"- {r}" for r in report["reasons"]] or ["- no retraining rule triggered"]
        lines += [f"- note: {n}" for n in report["notes"]]
        if report["psi"]:
            lines += ["", "| Feature | PSI | Status |", "|---|---|---|"]
            lines += [f"| {c} | {v:.4f} | {status(v)} |" for c, v in sorted(report["psi"].items(), key=lambda kv: -kv[1])]
        with open(summary_file, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


def status(value):
    return "shift" if value >= PSI_ALERT else ("watch" if value >= PSI_WATCH else "stable")


def main():
    parser = argparse.ArgumentParser(description="Decide whether to retrain the model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--deploy-dir", type=Path, default=DEFAULT_DEPLOY_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--force", action="store_true", help="retrain regardless of the rules")
    args = parser.parse_args()

    X = pd.read_csv(args.data_dir / "Xtrain.csv")
    y = pd.read_csv(args.data_dir / "ytrain.csv").squeeze("columns")
    profile_path = args.deploy_dir / PROFILE_FILE
    reference = json.loads(profile_path.read_text()) if profile_path.exists() else None
    data_sha = file_sha256(args.data_dir / "Xtrain.csv", args.data_dir / "ytrain.csv")
    code_sha = code_fingerprint()

    retrain, reasons, notes, psi_table, growth = decide(reference, X, y, data_sha, code_sha, args.force)
    report = {"retrain": retrain, "reasons": reasons, "notes": notes, "psi": psi_table,
              "n_current": int(len(X)), "n_reference": reference["n_rows"] if reference else None,
              "growth": growth, "data_changed": None if reference is None else data_sha != reference["train_data_sha256"],
              "code_changed": None if reference is None else code_sha != reference["code_sha256"]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    write_outputs(report)

    print("=" * 64 + "\nDRIFT CHECK\n" + "=" * 64)
    print(f"Live model reference : {'none' if reference is None else profile_path}")
    if reference is not None:
        print(f"Training customers   : {len(X):,} now vs {reference['n_rows']:,} at last training "
              f"({growth:+.1%})  |  data changed: {report['data_changed']}  |  code changed: {report['code_changed']}")
    if psi_table:
        table = pd.Series(psi_table, name="psi").sort_values(ascending=False).to_frame()
        table["status"] = table["psi"].map(status)
        print("\nPSI per model feature (current vs live model's training data)")
        print(table.to_string())
    print(f"\nDECISION: {'RETRAIN' if retrain else 'SKIP TRAINING'}")
    for r in reasons or ["no retraining rule triggered"]:
        print(f"  - {r}")
    for n in notes:
        print(f"  note: {n}")


if __name__ == "__main__":
    main()
