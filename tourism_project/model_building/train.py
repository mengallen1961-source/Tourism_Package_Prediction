# ==============================================================
# Cell 7.2 | Write the production training script
# --------------------------------------------------------------
# Purpose : Create train.py, the third job of the pipeline: tune the
#           configured recipe, choose the threshold, evaluate on the test
#           set, run the Champion/Challenger gate and save the model bundle.
# Input   : None (the script reads its inputs at run time, see below)
# Output  : tourism_project/model_building/train.py
# ==============================================================
"""
Train, evaluate and gate the tourism purchase model.

Input : artifacts/data/{Xtrain,Xtest,ytrain,ytest}.csv, split_manifest.csv  (artifact from data-prep)
        tourism_project/config/model_config.json                         (the recipe)
        <deploy-dir>/tourism_model.joblib + metrics.json                   (current champion, if any)
Output: <deploy-dir>/tourism_model.joblib, reference_profile.json, metrics.json
                                         (model and profile only when the challenger is deployed;
                                          the metrics.json history is always updated)
        artifacts/train_report.json, MLflow runs, GitHub Actions step summary
Exit  : 0 = finished (challenger deployed or rejected), 1 = error
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from drift_check import PROFILE_FILE, build_profile, code_fingerprint, file_sha256   # shared with the drift check

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "model_config.json"
DEFAULT_DEPLOY_DIR = PROJECT_ROOT / "deployment"
DEFAULT_DATA_DIR = Path("artifacts") / "data"
REPORT_PATH = Path("artifacts") / "train_report.json"
MODEL_FILE, METRICS_FILE = "tourism_model.joblib", "metrics.json"
EXPERIMENT = "tourism-production-training"
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


# ---------------------------------------------------------------- helpers
def build_estimator(name, y, seed):
    """Candidate algorithms, each weighting the minority class (as in development).
    Returns (estimator, fit-time keyword arguments)."""
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    weights = {"sample_weight": compute_sample_weight("balanced", y)}
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1), {}
    if name == "XGBoost":
        return xgb.XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss",
                                 random_state=seed, n_jobs=-1), {}
    if name == "Bagging":
        return BaggingClassifier(estimator=DecisionTreeClassifier(class_weight="balanced", random_state=seed),
                                 n_estimators=100, random_state=seed, n_jobs=-1), {}
    if name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=seed), weights
    if name == "AdaBoost":
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=200,
                                  random_state=seed), weights
    raise ValueError(f"Unknown algorithm in config: {name}")


def build_pipeline(features, X, estimator):
    """Scale numeric columns, one-hot encode text columns, then the model."""
    categorical = [f for f in features if X[f].dtype == object]
    numeric = [f for f in features if f not in categorical]
    preprocessor = make_column_transformer((StandardScaler(), numeric),
                                           (OneHotEncoder(handle_unknown="ignore"), categorical))
    return make_pipeline(preprocessor, estimator)


def input_schema(X):
    """What the app needs to build one input widget per feature."""
    schema = {}
    for col in X.columns:
        if X[col].dtype == object:
            schema[col] = {"type": "categorical", "levels": sorted(X[col].dropna().unique().tolist()),
                           "default": X[col].mode().iloc[0]}
        else:
            values = X[col].dropna()
            is_int = bool((values == values.round()).all())
            schema[col] = {"type": "numeric", "integer": is_int,
                           "min": float(values.min()), "max": float(values.max()),
                           "default": float(values.median())}
    return schema


def evaluate(proba, y, threshold):
    pred = (proba >= threshold).astype(int)
    return {"f1": f1_score(y, pred, zero_division=0), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "accuracy": accuracy_score(y, pred),
            "roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba),
            "flagged": int(pred.sum())}


def connect_mlflow():
    """Use the tracking server if it is up (GitHub Actions starts one); otherwise a local folder."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    try:
        ok = urllib.request.urlopen(f"{uri}/health", timeout=3).read().decode() == "OK"
    except Exception:
        ok = False
    if not ok:
        # separate folder: ./mlruns belongs to the server and uses server-side artifact URIs
        uri = Path("mlruns_local").resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)
    return uri


def gate(challenger, champion_bundle, champion_rescored, cfg_gate, strategy):
    """Champion/Challenger decision. Returns (deploy: bool, reason: str)."""
    if challenger["pr_auc"] < cfg_gate["min_pr_auc"] or challenger["f1"] < cfg_gate["min_f1"]:
        return False, (f"rejected: below quality floor (PR-AUC {challenger['pr_auc']:.4f} / F1 {challenger['f1']:.4f}"
                       f" vs minimum {cfg_gate['min_pr_auc']} / {cfg_gate['min_f1']})")
    if champion_bundle is None:
        return True, "deployed: first model (no champion yet)"
    if champion_bundle["strategy"] != strategy:
        return True, (f"deployed: strategy change {champion_bundle['strategy']} -> {strategy} "
                      f"(not comparable head-to-head; passes quality floor)")
    metric = cfg_gate["compare_metric"]
    delta = challenger[metric] - champion_rescored[metric]
    if delta >= -cfg_gate["tolerance"]:
        return True, (f"deployed: challenger {metric} {challenger[metric]:.4f} vs champion "
                      f"{champion_rescored[metric]:.4f} (change {delta:+.4f}, tolerance {cfg_gate['tolerance']})")
    return False, (f"rejected: challenger {metric} {challenger[metric]:.4f} is worse than champion "
                   f"{champion_rescored[metric]:.4f} by {-delta:.4f} (> tolerance {cfg_gate['tolerance']})")


def write_github_summary(report):
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return
    c, t = report["challenger"], report["challenger"]["test_metrics"]
    lines = ["## Model training and gate",
             f"**Decision:** {report['decision']}  ",
             f"**Strategy / feature set:** {c['strategy']} / {c['feature_set']} ({len(c['features'])} features)  ",
             f"**Algorithm:** {c['algorithm']}  |  **threshold:** {c['threshold']:.2f}  ",
             f"**Best parameters:** `{c['best_params']}`", "",
             "| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |", "|---|---|---|---|---|---|",
             f"| Challenger | {t['pr_auc']:.4f} | {t['roc_auc']:.4f} | {t['f1']:.4f} | {t['precision']:.4f} | {t['recall']:.4f} |"]
    if report.get("champion_rescored"):
        r = report["champion_rescored"]
        lines.append(f"| Champion (re-scored) | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['f1']:.4f} | "
                     f"{r['precision']:.4f} | {r['recall']:.4f} |")
    lines.append("")
    with open(summary_file, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Train and gate the tourism purchase model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--deploy-dir", type=Path, default=DEFAULT_DEPLOY_DIR)
    args = parser.parse_args()
    start = time.time()

    # 1. Recipe and data
    cfg = json.loads(args.config.read_text())
    features = cfg["feature_sets"][cfg["feature_set"]]
    seed = cfg["random_state"]
    X_train = pd.read_csv(args.data_dir / "Xtrain.csv")
    X_test = pd.read_csv(args.data_dir / "Xtest.csv")
    y_train = pd.read_csv(args.data_dir / "ytrain.csv").squeeze("columns")
    y_test = pd.read_csv(args.data_dir / "ytest.csv").squeeze("columns")
    missing = [f for f in features if f not in X_train.columns]
    if missing:
        raise ValueError(f"Configured features missing from the data: {missing}")
    fingerprints = {"train_data_sha256": file_sha256(args.data_dir / "Xtrain.csv", args.data_dir / "ytrain.csv"),
                    "test_data_sha256": file_sha256(args.data_dir / "Xtest.csv", args.data_dir / "ytest.csv")}

    # 2. Tune within the configured search space (objective: PR-AUC), logged to MLflow
    tracking_uri = connect_mlflow()
    estimator, fit_kwargs = build_estimator(cfg["algorithm"], y_train, seed)
    pipeline = build_pipeline(features, X_train, estimator)
    step = pipeline.steps[-1][0]                                   # e.g. "randomforestclassifier"
    grid = {f"{step}__{k}": v for k, v in cfg["search_space"].items()}
    fit_params = {f"{step}__{k}": v for k, v in fit_kwargs.items()}
    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)

    with mlflow.start_run(run_name=f"train | {cfg['algorithm']} | {cfg['feature_set']}") as run:
        mlflow.set_tags({"stage": "production-training", "strategy": cfg["strategy"],
                         "feature_set": cfg["feature_set"], "algorithm": cfg["algorithm"],
                         "github_sha": os.environ.get("GITHUB_SHA", "local")})
        mlflow.log_dict(cfg, "model_config.json")
        search = GridSearchCV(pipeline, grid, scoring=cfg["tuning_metric"], cv=cv, n_jobs=-1)
        search.fit(X_train[features], y_train, **fit_params)
        res = search.cv_results_
        for i, params in enumerate(res["params"]):                 # every tuned combination
            with mlflow.start_run(run_name=f"trial {i + 1:02d}", nested=True):
                mlflow.set_tag("stage", "production-tuning-trial")
                mlflow.log_params({k.split("__", 1)[1]: v for k, v in params.items()})
                mlflow.log_metrics({"cv_pr_auc_mean": res["mean_test_score"][i],
                                    "cv_pr_auc_std": res["std_test_score"][i]})
        best_params = {k.split("__", 1)[1]: v for k, v in search.best_params_.items()}

        # 3. Threshold from out-of-fold probabilities (test set untouched)
        oof = cross_val_predict(search.best_estimator_, X_train[features], y_train, cv=cv,
                                method="predict_proba", params=fit_params)[:, 1]
        oof_f1 = [f1_score(y_train, (oof >= t).astype(int), zero_division=0) for t in THRESHOLDS]
        threshold = float(THRESHOLDS[int(np.argmax(oof_f1))])

        # 4. Final challenger on the full training split, evaluated on the test split
        challenger_pipe = search.best_estimator_
        challenger_pipe.fit(X_train[features], y_train, **fit_params)
        test_metrics = evaluate(challenger_pipe.predict_proba(X_test[features])[:, 1], y_test, threshold)

        # 5. Champion / Challenger gate: re-score the champion on the SAME test customers
        model_path, metrics_path = args.deploy_dir / MODEL_FILE, args.deploy_dir / METRICS_FILE
        champion = joblib.load(model_path) if model_path.exists() else None
        champion_rescored = None
        if champion is not None:
            champion_rescored = evaluate(champion["pipeline"].predict_proba(X_test[champion["features"]])[:, 1],
                                         y_test, champion["threshold"])
        deploy, decision = gate(test_metrics, champion, champion_rescored, cfg["gate"], cfg["strategy"])

        mlflow.log_params({"algorithm": cfg["algorithm"], "feature_set": cfg["feature_set"],
                           "n_features": len(features), "threshold": threshold,
                           **{f"best_{k}": v for k, v in best_params.items()}})
        mlflow.log_metrics({"best_cv_pr_auc": search.best_score_, "oof_f1": max(oof_f1),
                            **{f"test_{k}": v for k, v in test_metrics.items()}})
        if champion_rescored:
            mlflow.log_metrics({f"champion_test_{k}": v for k, v in champion_rescored.items()})
        mlflow.set_tags({"gate_decision": "deployed" if deploy else "rejected", "gate_reason": decision})
        run_id = run.info.run_id

    # 6. Bundle, report card and history
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    challenger_info = {
        "strategy": cfg["strategy"], "feature_set": cfg["feature_set"], "features": features,
        "algorithm": cfg["algorithm"], "best_params": best_params, "threshold": threshold,
        "cv_best_pr_auc": float(search.best_score_), "oof_f1": float(max(oof_f1)),
        "test_metrics": test_metrics, "trained_at_utc": now, "mlflow_run_id": run_id,
        "git_sha": os.environ.get("GITHUB_SHA", "local"), **fingerprints, "code_sha256": code_fingerprint(),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                     "xgboost": xgb.__version__, "pandas": pd.__version__},
    }
    history = json.loads(metrics_path.read_text()).get("history", []) if metrics_path.exists() else []
    history.append({"time_utc": now, "decision": decision, "strategy": cfg["strategy"],
                    "feature_set": cfg["feature_set"], "test_pr_auc": round(test_metrics["pr_auc"], 4),
                    "test_f1": round(test_metrics["f1"], 4),
                    "champion_test_pr_auc": round(champion_rescored["pr_auc"], 4) if champion_rescored else None,
                    "champion_test_f1": round(champion_rescored["f1"], 4) if champion_rescored else None,
                    "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                    "git_sha": challenger_info["git_sha"]})
    args.deploy_dir.mkdir(parents=True, exist_ok=True)
    if deploy:
        bundle = {**challenger_info, "pipeline": challenger_pipe,
                  "input_schema": input_schema(X_train[features])}
        joblib.dump(bundle, model_path, compress=3)
        # Reference for future drift checks: profile of this training data + fingerprints
        profile = {**build_profile(X_train, y_train), "model_features": features,
                   "train_data_sha256": fingerprints["train_data_sha256"],
                   "code_sha256": challenger_info["code_sha256"], "trained_at_utc": now}
        (args.deploy_dir / PROFILE_FILE).write_text(json.dumps(profile, indent=2))
        live = challenger_info
    else:
        live = json.loads(metrics_path.read_text())["live_model"]
    metrics_path.write_text(json.dumps({"live_model": live, "last_decision": decision, "history": history},
                                       indent=2, default=str))

    report = {"decision": decision, "deployed": deploy, "challenger": challenger_info,
              "champion_rescored": champion_rescored, "mlflow_tracking_uri": tracking_uri,
              "runtime_seconds": round(time.time() - start, 1)}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    write_github_summary(report)

    # 7. Print
    line = "=" * 64
    print(f"{line}\nMODEL TRAINING AND GATE\n{line}")
    print(f"Strategy / feature set : {cfg['strategy']} / {cfg['feature_set']} ({len(features)} features)")
    print(f"Algorithm              : {cfg['algorithm']}  |  {len(res['params'])} combinations tuned")
    print(f"Best parameters        : {best_params}")
    print(f"Best CV PR-AUC         : {search.best_score_:.4f}  |  threshold {threshold:.2f} (OOF F1 {max(oof_f1):.4f})")
    table = {"challenger": test_metrics}
    if champion_rescored:
        table["champion (re-scored)"] = champion_rescored
    print("\nTest-set metrics")
    print(pd.DataFrame(table).T.round(4).astype({"flagged": int}).to_string())
    print(f"\nGATE: {decision}")
    print(f"Model bundle : {model_path} ({'updated' if deploy else 'unchanged'})")
    print(f"Report card  : {metrics_path}  |  MLflow: {tracking_uri}  |  {report['runtime_seconds']} s")


if __name__ == "__main__":
    main()
