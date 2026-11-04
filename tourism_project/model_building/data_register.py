# ==============================================================
# Cell 2.1 | Write the data registration script
# --------------------------------------------------------------
# Purpose : Create data_register.py, the first job of the pipeline.
#           (%%writefile must be the first line of a cell, so this header
#           is saved inside the script as well.)
# Input   : None (the script itself reads tourism_project/data/tourism.csv)
# Output  : tourism_project/model_building/data_register.py
# ==============================================================
"""
Register the raw tourism dataset stored in the repository.

Input : tourism_project/data/tourism.csv   (override with --data)
Output: console summary, artifacts/registration_report.json (override with --report)
        and, inside GitHub Actions, a Markdown summary on the run page.
Exit  : 0 = registered (warnings allowed), 1 = validation failed (pipeline stops)
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Data contract: what a valid tourism.csv must look like
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]          # .../tourism_project
DEFAULT_DATA = PROJECT_ROOT / "data" / "tourism.csv"
DEFAULT_REPORT = Path("artifacts") / "registration_report.json"

ID_COL, TARGET = "CustomerID", "ProdTaken"
MIN_ROWS = 500                                               # below this, training is not meaningful

NUMERIC_COLS = [
    "CustomerID", "ProdTaken", "Age", "CityTier", "DurationOfPitch",
    "NumberOfPersonVisiting", "NumberOfFollowups", "PreferredPropertyStar",
    "NumberOfTrips", "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_COLS = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]
EXPECTED_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# Known labels for each categorical column (anything else triggers a warning)
ALLOWED_CATEGORIES = {
    "TypeofContact": {"Self Enquiry", "Company Invited"},
    "Occupation": {"Salaried", "Small Business", "Large Business", "Free Lancer"},
    "Gender": {"Male", "Female"},
    "ProductPitched": {"Basic", "Standard", "Deluxe", "Super Deluxe", "King"},
    "MaritalStatus": {"Single", "Married", "Divorced", "Unmarried"},
    "Designation": {"Executive", "Manager", "Senior Manager", "AVP", "VP"},
}

# Copied records: the source data contains records that repeat an earlier customer
# (CustomerID - 1470) with at most 4 fields changed (evidence in Section 3.1).
NEAR_COPY_OFFSET = 1470
NEAR_COPY_MAX_DIFF = 4

# Plausible business ranges (inclusive) for numeric columns
VALUE_RANGES = {
    "Age": (18, 100), "CityTier": (1, 3), "DurationOfPitch": (0, 180),
    "NumberOfPersonVisiting": (1, 10), "NumberOfFollowups": (0, 10),
    "PreferredPropertyStar": (1, 5), "NumberOfTrips": (0, 50), "Passport": (0, 1),
    "PitchSatisfactionScore": (1, 5), "OwnCar": (0, 1),
    "NumberOfChildrenVisiting": (0, 10), "MonthlyIncome": (1, 1_000_000),
}


def sha256_of(path):
    """Return the SHA-256 fingerprint of a file (identifies the data version)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def near_copy_mask(df, feature_cols, offset=NEAR_COPY_OFFSET, max_diff=NEAR_COPY_MAX_DIFF):
    """Flag records that copy the record with ID = own ID - offset in all but <= max_diff features.

    Returns (mask, n_diff): a boolean Series (True = copy) and the number of differing
    features per record (NaN when no record with ID - offset exists).
    Reused by prep.py, so the rule is defined in one place only.
    """
    cols = [ID_COL] + list(feature_cols)
    originals = df[cols].copy()
    originals[ID_COL] = originals[ID_COL] + offset            # place each record at its copy's ID
    merged = df[cols].merge(originals, on=ID_COL, how="left", suffixes=("", "__orig"), indicator=True)
    has_original = (merged["_merge"] == "both").to_numpy()
    def differs(a, b):
        # Compare values, not text: the left join turns integer columns into floats (1 vs 1.0);
        # two missing values count as equal.
        return (a.ne(b) & ~(a.isna() & b.isna())).astype(int)
    n_diff = sum(differs(merged[c], merged[f"{c}__orig"]) for c in feature_cols).to_numpy()
    mask = pd.Series(has_original & (n_diff <= max_diff), index=df.index)
    return mask, pd.Series(n_diff, index=df.index).where(has_original)


def validate(df):
    """Check the dataframe against the data contract; return a list of check results."""
    checks = []

    def add(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})

    # 1. Size
    add("minimum rows", "PASS" if len(df) >= MIN_ROWS else "FAIL",
        f"{len(df):,} rows (minimum {MIN_ROWS})")

    # 2. Schema: expected columns present, unexpected columns flagged
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    add("expected columns", "FAIL" if missing else "PASS",
        f"missing: {missing}" if missing else f"all {len(EXPECTED_COLS)} present")
    extra = [c for c in df.columns if c not in EXPECTED_COLS]
    add("unexpected columns", "WARN" if extra else "PASS",
        f"{extra} (to be dropped in data preparation)" if extra else "none")
    if missing:
        return checks                                        # later checks need these columns

    # 3. Data types: numeric columns must be numeric
    non_numeric = [c for c in NUMERIC_COLS if not pd.api.types.is_numeric_dtype(df[c])]
    add("numeric data types", "FAIL" if non_numeric else "PASS",
        f"non-numeric: {non_numeric}" if non_numeric else "all numeric columns are numeric")

    # 4. Target: binary 0/1 with no missing values
    target_values = set(df[TARGET].dropna().unique())
    target_ok = target_values <= {0, 1} and df[TARGET].notna().all()
    add("target is binary", "PASS" if target_ok else "FAIL",
        f"values {sorted(int(v) for v in target_values)}, missing {int(df[TARGET].isna().sum())}")

    # 5. Identifier: unique per customer
    dup_ids = int(df[ID_COL].duplicated().sum())
    add("unique CustomerID", "FAIL" if dup_ids else "PASS", f"{dup_ids} duplicated IDs")

    # 6. Missing values in features
    na = df[EXPECTED_COLS].isna().sum()
    na = na[na > 0]
    add("missing values", "WARN" if len(na) else "PASS",
        na.to_dict() if len(na) else "no missing values")

    # 7. Unknown category labels
    unknown = {}
    for col, allowed in ALLOWED_CATEGORIES.items():
        counts = df[col].dropna().value_counts()
        bad = {str(k): int(v) for k, v in counts.items() if k not in allowed}
        if bad:
            unknown[col] = bad
    add("category labels", "WARN" if unknown else "PASS",
        unknown if unknown else "all labels recognised")

    # 8. Out-of-range numeric values
    out_of_range = {}
    if not non_numeric:
        for col, (low, high) in VALUE_RANGES.items():
            n_bad = int(((df[col] < low) | (df[col] > high)).sum())
            if n_bad:
                out_of_range[col] = n_bad
    add("value ranges", "WARN" if out_of_range else "PASS",
        out_of_range if out_of_range else "all values within business ranges")

    # 9. Duplicate customer records (identical apart from the ID)
    feature_cols = [c for c in EXPECTED_COLS if c != ID_COL]
    dup_rows = int(df.duplicated(subset=feature_cols).sum())
    add("duplicate records", "WARN" if dup_rows else "PASS", f"{dup_rows} duplicated records")

    # 10. Copied records: same customer re-entered under ID + 1470 with a few fields changed
    copies, _ = near_copy_mask(df, [c for c in EXPECTED_COLS if c not in (ID_COL, TARGET)])
    n_copies = int(copies.sum())
    add("copied records", "WARN" if n_copies else "PASS",
        f"{n_copies} records repeat the record with ID - {NEAR_COPY_OFFSET} "
        f"(<= {NEAR_COPY_MAX_DIFF} features changed)" if n_copies else "none")

    return checks


def class_balance(df):
    """Target distribution with an Overall total row."""
    table = df[TARGET].value_counts().sort_index().rename("count").to_frame()
    table.index = table.index.map({0: "0 - not purchased", 1: "1 - purchased"})
    table["share_%"] = (table["count"] / table["count"].sum() * 100).round(2)
    table.loc["Overall"] = [table["count"].sum(), 100.0]
    table["count"] = table["count"].astype(int)
    return table


def write_github_summary(report, balance):
    """Inside GitHub Actions, render the result on the workflow run's Summary page."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return
    lines = [
        "## Data registration",
        f"**Status:** {report['status']}  ",
        f"**Rows x columns:** {report['rows']:,} x {report['columns']}  ",
        f"**SHA-256:** `{report['sha256'][:16]}...`",
        "", "| Check | Status | Detail |", "|---|---|---|",
    ]
    lines += [f"| {c['check']} | {c['status']} | {c['detail']} |" for c in report["checks"]]
    lines += ["", "**Class balance**", "", "| Class | Count | Share % |", "|---|---|---|"]
    lines += [f"| {idx} | {int(row['count']):,} | {row['share_%']:.2f} |" for idx, row in balance.iterrows()]
    with open(summary_file, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Register the tourism dataset.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="path to the raw CSV")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="path of the JSON report")
    args = parser.parse_args()

    # Load
    if not args.data.exists():
        print(f"REGISTRATION FAILED: {args.data} not found.")
        sys.exit(1)
    df = pd.read_csv(args.data)

    # Validate
    checks = validate(df)
    n_fail = sum(c["status"] == "FAIL" for c in checks)
    n_warn = sum(c["status"] == "WARN" for c in checks)
    status = "FAILED" if n_fail else ("REGISTERED WITH WARNINGS" if n_warn else "REGISTERED")

    # Summarise
    balance = class_balance(df) if TARGET in df.columns and not n_fail else None
    overview = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "unique": df.nunique(),
    })

    report = {
        "dataset": str(args.data),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256_of(args.data),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "status": status,
        "checks": checks,
        "class_balance": balance.to_dict(orient="index") if balance is not None else None,
    }

    # Print
    line = "=" * 64
    print(line)
    print(f"DATA REGISTRATION: {status}")
    print(line)
    print(f"Dataset : {args.data}")
    print(f"Shape   : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"SHA-256 : {report['sha256']}")
    print("\nValidation checks")
    for c in checks:
        print(f"  [{c['status']:<4}] {c['check']:<20} {c['detail']}")
    print("\nColumn overview")
    print(overview.to_string())
    if balance is not None:
        print(f"\nTarget distribution ({TARGET})")
        print(balance.to_string())

    # Save the report (uploaded as a workflow artifact by GitHub Actions)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved to {args.report}")
    if balance is not None:
        write_github_summary(report, balance)

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
