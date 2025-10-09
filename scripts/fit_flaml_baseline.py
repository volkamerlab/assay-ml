import os
import sys
import pandas as pd
import numpy as np
import joblib
from flaml import AutoML
from tqdm.auto import tqdm
from dti.featurization import MolFingerprint

CHUNK_ID = int(sys.argv[1])
log = lambda message: print(f"[{CHUNK_ID}] {message}")
CHUNK_FILE = f"data/raw/chembl_split_{CHUNK_ID}.csv"
MODEL_DIR = "data/flaml"
os.makedirs(MODEL_DIR, exist_ok=True)

log(f"reading csv")
df = pd.read_csv(CHUNK_FILE)

fpgen = MolFingerprint("morgan")
df["y_pred"] = np.nan
df["model_path"] = None


def train_assay(assay_id, group):
    log(f"processing assay {assay_id}")
    smiles_all = group["canonical_smiles"].values
    fps_all = np.stack(fpgen.compute_parallel(smiles_all, pbar=False))

    X_train = fps_all[~group["test"].values]
    X_test = fps_all[group["test"].values]

    y_train = group.loc[~group["test"], "target_transformed"].values

    if len(y_train) < 5 or X_test.shape[0] < 1:
        return assay_id, None, None, None

    try:
        n_features = X_train.shape[1]
        X_train = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(n_features)])
        X_test = pd.DataFrame(X_test, columns=[f"f{i}" for i in range(n_features)])

        automl = AutoML()
        automl_settings = {
            "time_budget": 60,
            "task": "regression",
            "metric": "rmse",
            "log_file_name": f"{MODEL_DIR}/assay_{assay_id}.log",
            "verbose": 3,
            "n_jobs": 16,
        }

        automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
        y_pred = automl.predict(X_test)

        model_path = os.path.join(MODEL_DIR, f"assay_{assay_id}.pkl")
        joblib.dump(automl, model_path)
    except:
        return assay_id, None, None, None

    return assay_id, group.loc[group["test"]].index, y_pred, model_path


results = [train_assay(assay_id, group) for assay_id, group in df.groupby("assay_id")]

log("write results")
for assay_id, test_idx, y_pred, model_path in results:
    if test_idx is None:
        continue
    df.loc[test_idx, "y_pred"] = y_pred
    df.loc[df["assay_id"] == assay_id, "model_path"] = model_path

df.to_csv(f"data/flaml/chembl_endpoints_split_flaml_chunk_{CHUNK_ID}.csv", index=False)
