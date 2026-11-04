# ==============================================================
# Cell 4.1 | Write the data preparation script
# --------------------------------------------------------------
# Purpose : Create prep.py, the second job of the pipeline: clean the
#           registered data, split it into a fixed ID-based train/test
#           split and save the splits locally.
# Input   : None (the script reads tourism_project/data/tourism.csv)
# Output  : tourism_project/model_building/prep.py
# ==============================================================
"""
Prepare the registered tourism data for modelling.

Input : tourism_project/data/tourism.csv          (override with --data)
Output: <out-dir>/Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv
        <out-dir>/split_manifest.csv              (CustomerID -> train/test)
        artifacts/prep_report.json                (cleaning audit and split checks)
        default <out-dir> = artifacts/data; passed to the next job as a workflow artifact
Exit  : 0 = success, 1 = a split check failed (pipeline stops)
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

from data_register import near_copy_mask             # same copied-record rule as registration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "tourism.csv"
DEFAULT_OUT = Path("artifacts") / "data"
REPORT_PATH = Path("artifacts") / "prep_report.json"

ID_COL, TARGET = "CustomerID", "ProdTaken"
DROP_COLUMNS = ["Unnamed: 0"]                         # leftover index column
LABEL_FIXES = {"Gender": {"Fe Male": "Female"}}       # known label typos
TEST_PERCENT = 20                                     # share of customers held out for testing
HOLDOUT_SALT = "tourism-holdout-v1"                   # changing it would define a NEW test set


def clean(df):
    """Apply the cleaning rules; return the clean frame and an audit trail."""
    audit = [{"step": "raw data", "rows": len(df), "columns": df.shape[1]}]

    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    audit.append({"step": f"drop columns {DROP_COLUMNS}", "rows": len(df), "columns": df.shape[1]})

    for col, mapping in LABEL_FIXES.items():
        n_fixed = int(df[col].isin(mapping.keys()).sum())
        df[col] = df[col].replace(mapping)
        audit.append({"step": f"fix labels in {col} ({n_fixed} values)", "rows": len(df), "columns": df.shape[1]})

    feature_cols = [c for c in df.columns if c != ID_COL]
    df = df.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)
    audit.append({"step": "drop exact duplicates", "rows": len(df), "columns": df.shape[1]})

    copies, _ = near_copy_mask(df, [c for c in feature_cols if c != TARGET])
    df = df[~copies].reset_index(drop=True)
    audit.append({"step": "drop copied records (ID - 1470, <= 4 fields changed)", "rows": len(df), "columns": df.shape[1]})
    return df, audit


def assign_split(customer_ids):
    """Deterministic split: hash each ID to 0-99; values below TEST_PERCENT -> test."""
    def bucket(customer_id):
        digest = hashlib.sha256(f"{HOLDOUT_SALT}:{customer_id}".encode()).hexdigest()
        return int(digest, 16) % 100
    return pd.Series(["test" if bucket(i) < TEST_PERCENT else "train" for i in customer_ids],
                     index=customer_ids.index, name="split")


def write_github_summary(report):
    """Inside GitHub Actions, add the preparation result to the run's Summary page."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return
    lines = ["## Data preparation", "", "| Step | Rows | Columns |", "|---|---|---|"]
    lines += [f"| {a['step']} | {a['rows']:,} | {a['columns']} |" for a in report["cleaning_audit"]]
    lines += ["", "| Split | Customers | Buyers | Conversion % |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['customers']:,} | {v['buyers']:,} | {v['conversion_%']:.2f} |"
              for k, v in report["split_summary"].items()]
    lines.append("")
    with open(summary_file, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Clean and split the tourism dataset.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    # 1. Load and clean
    df, audit = clean(pd.read_csv(args.data))

    # 2. Fixed, ID-based train/test split
    split = assign_split(df[ID_COL])
    train, test = df[split == "train"], df[split == "test"]

    # 3. Separate features and target (CustomerID is not a feature)
    X_train, y_train = train.drop(columns=[ID_COL, TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[ID_COL, TARGET]), test[TARGET]

    # 4. Sanity checks: both classes present in both splits, no customer in both
    checks = {
        "both classes in train": bool(y_train.nunique() == 2),
        "both classes in test": bool(y_test.nunique() == 2),
        "no customer in both splits": bool(set(train[ID_COL]).isdisjoint(test[ID_COL])),
        "test share between 15% and 25%": bool(0.15 <= len(test) / len(df) <= 0.25),
    }

    # 5. Save the splits and the manifest
    args.out_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(args.out_dir / "Xtrain.csv", index=False)
    X_test.to_csv(args.out_dir / "Xtest.csv", index=False)
    y_train.to_csv(args.out_dir / "ytrain.csv", index=False)
    y_test.to_csv(args.out_dir / "ytest.csv", index=False)
    pd.DataFrame({ID_COL: df[ID_COL], "split": split}).to_csv(args.out_dir / "split_manifest.csv", index=False)

    # 6. Report (split summary includes an Overall row)
    def summary(y):
        return {"customers": int(len(y)), "buyers": int(y.sum()), "conversion_%": round(float(y.mean() * 100), 2)}
    report = {
        "cleaning_audit": audit,
        "split_summary": {"train": summary(y_train), "test": summary(y_test), "Overall": summary(df[TARGET])},
        "n_features": int(X_train.shape[1]),
        "features": list(X_train.columns),
        "checks": checks,
        "holdout_rule": f"sha256('{HOLDOUT_SALT}:<CustomerID>') % 100 < {TEST_PERCENT}",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    write_github_summary(report)

    # 7. Print
    print("=" * 64 + "\nDATA PREPARATION\n" + "=" * 64)
    print(pd.DataFrame(audit).set_index("step").to_string())
    print("\nSplit summary")
    print(pd.DataFrame(report["split_summary"]).T.astype({"customers": int, "buyers": int}).to_string())
    print(f"\nFeatures kept ({X_train.shape[1]}): {list(X_train.columns)}")
    print("\nChecks")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nSplits written to {args.out_dir}/  |  report: {REPORT_PATH}")

    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
