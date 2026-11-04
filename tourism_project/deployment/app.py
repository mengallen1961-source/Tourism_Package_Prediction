# ==============================================================
# Cell 8.2 | Write the Streamlit app
# --------------------------------------------------------------
# Purpose : Create the web app served by Streamlit Community Cloud: a form
#           generated from the model's input schema, the inputs collected
#           in a DataFrame, a prediction with the model's own threshold,
#           batch scoring of a CSV and a model card in the sidebar.
# Input   : None (the app reads tourism_model.joblib and metrics.json next
#           to it, or from the folder in env var TOURISM_MODEL_DIR)
# Output  : tourism_project/deployment/app.py
# ==============================================================
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from scoring import load_bundle, load_report_card, out_of_range, rank_call_list, score, validate

MODEL_DIR = Path(os.environ.get("TOURISM_MODEL_DIR", Path(__file__).resolve().parent))

LABELS = {   # friendly labels and help texts; unknown features fall back to their column name
    "Age": ("Age", "Customer's age in years"),
    "TypeofContact": ("Type of contact", "How the lead arrived"),
    "CityTier": ("City tier", "1 = most developed city category, 3 = least"),
    "Occupation": ("Occupation", None),
    "Gender": ("Gender", None),
    "NumberOfPersonVisiting": ("Persons travelling", "Total number of people on the trip"),
    "PreferredPropertyStar": ("Preferred hotel rating", "Stars"),
    "MaritalStatus": ("Marital status", None),
    "NumberOfTrips": ("Trips per year", "Average number of trips per year"),
    "Passport": ("Holds a passport", None),
    "OwnCar": ("Owns a car", None),
    "NumberOfChildrenVisiting": ("Children under 5 travelling", None),
    "Designation": ("Designation", "Job level in the current organisation"),
    "MonthlyIncome": ("Monthly income", "Gross monthly income"),
    "ProductPitched": ("Product pitched", None),
    "DurationOfPitch": ("Pitch duration (minutes)", "Known only after the first contact"),
    "NumberOfFollowups": ("Follow-ups", "Known only after the first contact"),
    "PitchSatisfactionScore": ("Pitch satisfaction (1-5)", "Known only after the first contact"),
}
YES_NO = {"Passport", "OwnCar"}

st.set_page_config(page_title="Wellness Package Purchase Prediction", layout="wide")


@st.cache_resource
def get_bundle(model_version):
    # Cached per file version: when the pipeline commits a new model, its modification time changes
    # and the new bundle is loaded instead of the cached one.
    return load_bundle(MODEL_DIR)


model_file = MODEL_DIR / "tourism_model.joblib"
bundle = get_bundle(model_file.stat().st_mtime if model_file.exists() else None)
card = load_report_card(MODEL_DIR)
st.title("Wellness Tourism Package - Purchase Prediction")
st.caption("Visit with Us | scores customers before the first sales call")

if bundle is None:
    st.warning("No model has been deployed yet. The GitHub Actions pipeline commits the model here after its first run.")
    st.stop()

# ---------------- sidebar: model card
with st.sidebar:
    st.header("Live model")
    st.write(f"**Strategy:** {bundle['strategy']}  \n**Algorithm:** {bundle['algorithm']}  \n"
             f"**Features:** {len(bundle['features'])}  \n**Decision threshold:** {bundle['threshold']:.2f}")
    m = bundle["test_metrics"]
    c1, c2 = st.columns(2)
    c1.metric("Test F1", f"{m['f1']:.3f}")
    c2.metric("Test PR-AUC", f"{m['pr_auc']:.3f}")
    c1.metric("Precision", f"{m['precision']:.3f}")
    c2.metric("Recall", f"{m['recall']:.3f}")
    st.caption(f"Trained {bundle['trained_at_utc']} on {bundle['n_train']:,} customers | "
               f"commit {str(bundle.get('git_sha', 'local'))[:7]} | Python {bundle['versions']['python']}")
    if card:
        st.caption(f"Last pipeline decision: {card['last_decision']}")

single, batch = st.tabs(["Score one customer", "Score a customer list (CSV)"])

# ---------------- single customer: form generated from the input schema
with single:
    with st.form("customer"):
        cols = st.columns(3)
        values = {}
        for i, feature in enumerate(bundle["features"]):
            spec = bundle["input_schema"][feature]
            label, help_text = LABELS.get(feature, (feature, None))
            box = cols[i % 3]
            if feature in YES_NO:
                values[feature] = 1 if box.radio(label, ["No", "Yes"], horizontal=True, help=help_text) == "Yes" else 0
            elif spec["type"] == "categorical":
                levels = spec["levels"]
                values[feature] = box.selectbox(label, levels, index=levels.index(spec["default"]), help=help_text)
            elif spec["integer"] and spec["max"] - spec["min"] <= 60:          # e.g. age, trips, stars
                values[feature] = box.slider(label, int(spec["min"]), int(spec["max"]), int(spec["default"]), help=help_text)
            elif spec["integer"]:                                                 # e.g. monthly income
                values[feature] = box.number_input(label, min_value=0, value=int(spec["default"]),
                                                   step=500 if spec["max"] > 1000 else 1, help=help_text)
            else:
                values[feature] = box.number_input(label, min_value=0.0, value=float(spec["default"]), help=help_text)
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        input_df = pd.DataFrame([values])                     # the inputs, saved into a dataframe
        st.subheader("Input received by the model")
        st.dataframe(input_df, hide_index=True, use_container_width=True)
        for note in out_of_range(bundle, input_df):           # only free-typed numbers can leave the range
            st.warning(note)
        result = score(bundle, input_df).iloc[0]
        p, t = result["purchase_probability"], bundle["threshold"]
        st.subheader("Prediction")
        if result["recommendation"] == "call":
            st.success(f"Likely buyer - recommend a call. Purchase probability {p:.1%} (threshold {t:.0%}).")
        else:
            st.info(f"Unlikely buyer - lower priority. Purchase probability {p:.1%} (threshold {t:.0%}).")
        st.progress(float(min(p, 1.0)))

# ---------------- batch: CSV -> ranked call list
with batch:
    st.write("Upload a CSV with one row per customer and the columns: " + ", ".join(f"`{f}`" for f in bundle["features"]))
    upload = st.file_uploader("Customer list", type="csv")
    if upload is not None:
        customers = pd.read_csv(upload)
        problems = validate(bundle, customers)
        if problems:
            st.error("The file cannot be scored:\n\n" + "\n".join(f"- {p}" for p in problems))
        else:
            for note in out_of_range(bundle, customers):
                st.warning(note)
            ranked = rank_call_list(bundle, customers)
            n_call = int((ranked["recommendation"] == "call").sum())
            st.success(f"{n_call} of {len(ranked)} customers recommended for a call "
                       f"({n_call / len(ranked):.0%}); sorted from most to least likely buyer.")
            st.dataframe(ranked, hide_index=True, use_container_width=True)
            st.download_button("Download call list", ranked.to_csv(index=False), "call_list.csv", "text/csv")
