# Visit with Us - Wellness Tourism Package Purchase Prediction

End-to-end MLOps project: a model that predicts whether a customer will buy the Wellness Tourism
Package **before the first sales call**, trained, gated and deployed automatically by GitHub Actions
and served with Streamlit Community Cloud.

**Live app:** deployed in Section 11 of the notebook

## Repository structure
```
.github/workflows/pipeline.yml        CI/CD/CT pipeline (register -> prepare -> drift check -> train/gate/commit)
tourism_project/
  data/tourism.csv                    registered raw dataset
  model_building/
    data_register.py                  data contract validation + fingerprint
    prep.py                           cleaning (incl. copied records) + fixed ID-based train/test split
    drift_check.py                    retrain decision: new data / PSI drift / code or config change
    train.py                          tuning + MLflow tracking + threshold + Champion/Challenger gate
  config/model_config.json            the recipe (strategy, features, algorithm, search space, gate)
  deployment/
    app.py, scoring.py                Streamlit app and scoring logic
    requirements.txt                  app dependencies (Python 3.11)
    tourism_model.joblib              live model bundle      (committed by the pipeline)
    metrics.json                      report card + history  (committed by the pipeline)
    reference_profile.json            drift reference        (committed by the pipeline)
  requirements.txt                    pipeline dependencies (Python 3.11)
```

## Pipeline
| Job | What it does |
|---|---|
| register-dataset | validates the CSV against the data contract (fails the run on bad data) |
| data-prep | cleans the data and writes a fixed, ID-based train/test split (artifact) |
| drift-check | decides whether retraining is needed (outputs `retrain=true/false`) |
| model-training | runs only if needed: tunes, logs to MLflow, evaluates, gates and commits the model |

Triggered by every push to `main` that changes `tourism_project/` or the workflow, or manually
(Actions -> Tourism Project Pipeline -> Run workflow, optionally with *force_retrain*).
