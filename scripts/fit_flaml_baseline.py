import pandas as pd
import os
import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from flaml import AutoML
from tqdm.auto import tqdm

from dti.featurization import MolFingerprint
from dti.constants import ASSAY, SMILES, ACT

df = pd.read_csv("data/raw/chembl_endpoints_split.csv", index_col=False)

MODEL_DIR = "chembl_flaml"
os.makedirs(MODEL_DIR, exist_ok=True)

fpgen = MolFingerprint("morgan")

df["y_pred"] = np.nan
df["model_path"] = None


def train_assay(assay_id, group):
    X_train = np.stack(
        fpgen.compute_parallel(group.loc[~group["test"], "smiles"].values, pbar=False)
    )
    X_test = np.stack(fpgen.compute_parallel(group.loc[group["test"], "smiles"].values))

    y_train = group.loc[~group["test"], "activity_value"].values
    y_test = group.loc[group["test"], "activity_value"].values

    if len(y_train) < 5 or len(y_test) < 1:
        return assay_id, None, None, None  # skip

    automl = AutoML()
    automl_settings = {
        "time_budget": 60,
        "task": "regression",
        "metric": "rmse",
        "log_file_name": f"{MODEL_DIR}/assay_{assay_id}.log",
        "verbose": 0,
    }

    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
    y_pred = automl.predict(X_test)

    model_path = os.path.join(MODEL_DIR, f"assay_{assay_id}.pkl")
    joblib.dump(automl, model_path)

    return assay_id, group.loc[group["test"]].index, y_pred, model_path


# Run in parallel across assays
results = Parallel(n_jobs=-1)(  # set n_jobs=N for #CPUs you want
    delayed(train_assay)(assay_id, group)
    for assay_id, group in tqdm(df.groupby("assay_id"), total=df["assay_id"].nunique())
)

# Write results back into df
for assay_id, test_idx, y_pred, model_path in results:
    if test_idx is None:
        continue
    df.loc[test_idx, "y_pred"] = y_pred
    df.loc[df["assay_id"] == assay_id, "model_path"] = model_path

df.to_csv("data/raw/chembl_endpoints_split_flaml.csv", index_col=False)
