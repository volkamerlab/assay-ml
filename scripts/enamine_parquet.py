import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm.auto as tqdm

from dti.featurization import MolFingerprint

from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def compute_props(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    props = Descriptors.CalcMolDescriptors(mol)
    return props


def preprocess_csv_to_parquet(csv_path, parquet_path, nbits=2048, chunksize=100_000):
    reader = pd.read_csv(csv_path, chunksize=chunksize)
    fp = MolFingerprint("morgan")
    writer = None
    n_workers = multiprocessing.cpu_count() + 3

    for i, chunk in tqdm.tqdm(enumerate(reader)):
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(
                executor.map(compute_props, chunk["SMILES"], [nbits] * len(chunk))
            )

        records = [
            {"smiles": smi, **props}
            for smi, props in zip(chunk["SMILES"], results)
            if props is not None
        ]

        if not records:
            continue

        df = pd.DataFrame(records)

        # compute fingerprints (assuming this is already parallelized inside)
        df["fp"] = fp.compute_parallel(df["smiles"])

        arrays = {k: pa.array(df[k]) for k in df.columns if k != "fp"}
        arrays["fp"] = pa.FixedSizeListArray.from_arrays(
            np.stack(df["fp"].values).ravel(), list_size=nbits
        )
        table = pa.table(arrays)

        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema, compression="zstd")
        writer.write_table(table)
        print(f"Processed chunk {i}")

    if writer:
        writer.close()


def main():
    preprocess_csv_to_parquet(
        "../data/raw/cleaned_enamine.csv.gz", "../data/raw/cleaned_enamine.parquet"
    )


if __name__ == "__main__":
    main()
