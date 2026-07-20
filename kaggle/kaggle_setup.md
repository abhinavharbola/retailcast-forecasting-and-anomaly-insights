# Kaggle Setup:

## 1. Create the notebook
1. Go to the competition dataset page: **Store Sales - Time Series Forecasting** (Kaggle).
2. Click "New Notebook" from that dataset page, this auto-attaches the dataset as input at
   `/kaggle/input/competetions/store-sales-time-series-forecasting/`.
3. Settings (right panel): Accelerator = **None/CPU** (no GPU needed anywhere in this project).
4. Settings → Internet = **ON** (needed to `pip install` prophet/lightgbm/xgboost/mlflow/dagshub).

## 2. Install packages (first cell of every notebook)
```python
!pip install -q prophet statsmodels lightgbm xgboost mlflow dagshub scikit-learn pyarrow
```
(`numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn` are preinstalled on Kaggle, no need to reinstall.)

## 3. DagsHub + MLflow remote tracking (needed from Notebook 4 onward)
1. You already have a DagsHub account from your MLOps project, create a **new repo** there for RetailCast (or reuse the same one with a different experiment name, your call).
2. Get your DagsHub token: DagsHub → Settings → Tokens.
3. In Kaggle: Add-ons → Secrets → add two secrets:
   - `DAGSHUB_TOKEN` = your token
   - `DAGSHUB_REPO` = `yourusername/retailcast` (or whatever repo name you use)
4. Each notebook that logs to MLflow will read these secrets, code included in Notebook 4.

## 4. LLM API keys (needed later, in local dashboard stage, not needed for Notebooks 1-5)
Not required yet. We'll set these up when we get to the GenAI narrative layer.

## 5. Order of execution
Run these notebooks **in order**, each one saves its output to `/kaggle/working/` as the input for the next:
1. `01_eda.py` → saves `retailcast_subset.parquet`
2. `02_feature_engineering.py` → saves `retailcast_features.parquet`
3. `03_statistical_models.py` → saves `statistical_results.csv`
4. `04_ml_models.py` → saves `ml_results.csv`, `final_holdout_predictions.parquet`, logs to MLflow/DagsHub
5. `05_anomaly_detection.py` → saves `anomaly_results.parquet`, `anomaly_eval_metrics.csv`

**Important**: Kaggle notebook sessions are ephemeral, at the end of each notebook, go to
"Save Version" → "Save & Run All" so outputs persist, then download the output files
(top-right "Output" tab) to keep locally, since the next notebook needs them as input
(upload the previous notebook's output as a new Kaggle Dataset, or attach the previous
notebook itself as an input source via "Add Data" → "Notebook Output").