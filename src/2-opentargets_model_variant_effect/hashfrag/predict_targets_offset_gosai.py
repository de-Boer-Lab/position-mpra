from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
import argparse
import sys

sys.path.append(
    "/scratch/st-cdeboer-1/sambina/mpra/models/random-promoter-dream-challenge-2022/benchmarks/human"
)

FASTA_PATH = "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/2-opentargets_model_variant_effect/variants_200bp_offsets_hashfrag_filtered.fa"
K562_PATH = "/scratch/st-cdeboer-1/sambina/mpra/data/chromosome/gosai/data_lfcse/data_k562/fold_0/valid.txt.gz"


def parse_fasta(fasta_path):
    """Parse FASTA into {(variant_id, offset): {'ref': seq, 'alt': seq}}."""
    records = {}
    current_key = None
    current_allele = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_key is not None:
                    records.setdefault(current_key, {})[current_allele] = "".join(current_seq)
                header = line[1:]
                variant_id, offset_allele = header.split("_offset")
                offset, allele = offset_allele.rsplit("_", 1)
                current_key = (variant_id, int(offset))
                current_allele = allele
                current_seq = []
            else:
                current_seq.append(line)

    if current_key is not None:
        records.setdefault(current_key, {})[current_allele] = "".join(current_seq)

    return records


def one_hot_encode(seq):
    mapping = {"A": [1, 0, 0, 0], "G": [0, 1, 0, 0], "C": [0, 0, 1, 0], "T": [0, 0, 0, 1]}
    return [mapping.get(base, [0, 0, 0, 0]) for base in seq.upper()]


def encode_for_model(seq, seq_size):
    # 5th channel = reverse strand indicator, set to 0
    encoded = [base + [0] for base in one_hot_encode(seq)]
    arr = np.array(encoded, dtype=np.float32).reshape(1, seq_size, 5).transpose(0, 2, 1)
    return torch.tensor(arr, dtype=torch.float32)


def main(model_path, output_path, fasta_path=FASTA_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from prixfixe.autosome import AutosomeFinalLayersBlock
    from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
    from prixfixe.prixfixe import PrixFixeNet

    SEQ_SIZE = 231
    generator = torch.Generator()

    first = BHIFirstLayersBlock(
        in_channels=5,
        out_channels=320,
        seqsize=SEQ_SIZE,
        kernel_sizes=[9, 15],
        pool_size=1,
        dropout=0.2,
    )
    core = BHICoreBlock(
        in_channels=first.out_channels,
        out_channels=320,
        seqsize=first.infer_outseqsize(),
        lstm_hidden_channels=320,
        kernel_sizes=[9, 15],
        pool_size=1,
        dropout1=0.2,
        dropout2=0.5,
    )
    final = AutosomeFinalLayersBlock(in_channels=core.out_channels)

    model = PrixFixeNet(first=first, core=core, final=final, generator=generator)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 15 bp upstream + 200 bp FASTA seq + 16 bp downstream = 231 bp
    k562 = pd.read_csv(K562_PATH, sep="\t", compression="gzip", nrows=1)
    upstream = k562.iloc[0]["seq"][:15]
    downstream = k562.iloc[0]["seq"][-16:]
    assert len(upstream) + 200 + len(downstream) == SEQ_SIZE, (
        f"Flank lengths don't sum to {SEQ_SIZE}: {len(upstream)} + 200 + {len(downstream)}"
    )

    records = parse_fasta(fasta_path)
    print(f"Loaded {len(records)} (variant, offset) pairs from FASTA")

    results = []
    for (variant_id, offset), alleles in tqdm(records.items()):
        if "ref" not in alleles or "alt" not in alleles:
            print(f"Warning: missing ref or alt for {variant_id} offset {offset}, skipping")
            continue

        ref_seq = upstream + alleles["ref"] + downstream
        alt_seq = upstream + alleles["alt"] + downstream

        if len(ref_seq) != SEQ_SIZE:
            print(
                f"Warning: sequence length {len(ref_seq)} != {SEQ_SIZE} for {variant_id} offset {offset}, skipping"
            )
            continue

        ref_tensor = encode_for_model(ref_seq, SEQ_SIZE).to(device)
        alt_tensor = encode_for_model(alt_seq, SEQ_SIZE).to(device)

        with torch.no_grad():
            pred_ref = model(ref_tensor).cpu().flatten().tolist()
            pred_alt = model(alt_tensor).cpu().flatten().tolist()

        pred_diff = [r - a for r, a in zip(pred_ref, pred_alt)]

        results.append(
            {
                "variant_id": variant_id,
                "offset": offset,
                "ref_pred": pred_ref,
                "alt_pred": pred_alt,
                "ref_minus_alt": pred_diff,
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values(["variant_id", "offset"]).reset_index(drop=True)
    df.to_csv(output_path, index=False, compression="gzip")
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Gosai model predictions on offset FASTA sequences."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output CSV.gz")
    parser.add_argument(
        "--fasta_path",
        type=str,
        default=FASTA_PATH,
        help="Path to input FASTA (default: hashfrag-filtered offsets)",
    )
    args = parser.parse_args()

    main(args.model_path, args.output_path, args.fasta_path)
