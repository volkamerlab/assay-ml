# https://www.kaggle.com/code/viktorfairuschin/extracting-esm-2-embeddings-from-fasta-files
import pathlib
import torch
import tqdm
import pandas as pd

from esm import FastaBatchedDataset, pretrained


def extract_embeddings(
    model_name,
    fasta_file,
    output_dir,
    tokens_per_batch=4096,
    seq_length=5000,
    repr_layers=[33],
):
    print("Setting up model")
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(seq_length),
        batch_sampler=batches,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing batch {batch_idx + 1} of {len(batches)}")

            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                entry_id = label.split()[0]

                filename = output_dir / f"{entry_id}.pt"
                truncate_len = min(seq_length, len(strs[i]))

                result = {"entry_id": entry_id}
                result["representation"] = {
                    layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                    for layer, t in representations.items()
                }

                torch.save(result, filename)


def write_kinodata_fasta():
    print("Reading kinodata activities")
    kd = pd.read_csv("data/raw/activities-chembl33_v0.5.csv", index_col=0)
    done = []
    print("Writing protein sequences")
    with open("data/data.fasta", "w") as f:
        for i, row in tqdm.tqdm(kd.iterrows(), total=len(kd)):
            uniprot = row["UniprotID"]
            if uniprot in done:
                continue
            f.write(f">{uniprot}\n{row['component_sequences.sequence']}\n")
            done.append(uniprot)


if __name__ == "__main__":
    write_kinodata_fasta()
    model_name = "esm2_t33_650M_UR50D"
    fasta_file = pathlib.Path("data/data.fasta")
    output_dir = pathlib.Path("data/esm_embeddings")
    output_dir.mkdir(exist_ok=True)
    extract_embeddings(model_name, fasta_file, output_dir)
