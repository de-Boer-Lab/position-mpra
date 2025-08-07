from Bio import SeqIO
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
import argparse
import sys
from Bio import SeqIO

sys.path.append("/scratch/st-cdeboer-1/sambina/mpra/mpra_models/random-promoter-dream-challenge-2022/benchmarks/human")

def main(model_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snvs_only = pd.read_csv("/scratch/st-cdeboer-1/sambina/position_mpra/data/opentargets_model_data/snvs_only_gwas_0.1_postprob.csv")
    genome_path = "/scratch/st-cdeboer-1/sambina/reference_genome/hg38.fa"

    with open(genome_path, "rt") as handle:
        genome = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))
        
    from prixfixe.autosome import AutosomeFinalLayersBlock
    from prixfixe.bhi import BHIFirstLayersBlock, BHICoreBlock
    from prixfixe.prixfixe import PrixFixeNet

    SEQ_SIZE = 231
    sidemer = 100 
    generator = torch.Generator()


    first = BHIFirstLayersBlock(in_channels=5, out_channels=320, seqsize=SEQ_SIZE, kernel_sizes=[9, 15], pool_size=1, dropout=0.2)
    core = BHICoreBlock(in_channels=first.out_channels, out_channels=320, seqsize=first.infer_outseqsize(), lstm_hidden_channels=320, kernel_sizes=[9, 15], pool_size=1, dropout1=0.2, dropout2=0.5)
    final = AutosomeFinalLayersBlock(in_channels=core.out_channels)

    model_rnn = PrixFixeNet(first=first, core=core, final=final, generator=generator)
    # model_rnn.load_state_dict(torch.load(f"{model_path}", map_location=device))
    model_rnn = model_rnn.to(device)
    state_dict = torch.load(model_path, map_location=device)
    model_rnn.load_state_dict(state_dict)
    model_rnn.eval()


    def one_hot_encode(seq):
        mapping = {'A': [1, 0, 0, 0], 'G': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
        return [mapping.get(base, [0, 0, 0, 0]) for base in seq]


    def generate_sequences(row, sidemer_offset=0):
        chrom = "chr" + str(row['CHR_ID'])
        pos = int(row['Position']) - 1 
        ref_nucleotide = row['Ref']
        alt_nucleotide = row['Alt']
        
        sequence_upstream = genome[chrom].seq[pos - sidemer + sidemer_offset: pos].upper()
        sequence_downstream = genome[chrom].seq[pos + 1: pos + sidemer + sidemer_offset].upper()
        
        ref_sequence = sequence_upstream + ref_nucleotide + sequence_downstream
        # print(len(ref_sequence))
        alt_sequence = sequence_upstream + alt_nucleotide + sequence_downstream
        if ref_sequence == alt_sequence:
            print(f"Assertion failed at position {row['CHR_ID']}:{row['Position']} with ref {row['Ref']} and alt {row['Alt']}")
            assert(ref_sequence != alt_sequence)
        return {'Ref_Sequence': str(ref_sequence), 'Alt_Sequence': str(alt_sequence)}


    def make_predictions(info_list, upstream, downstream):
        predictions = []
        for row in tqdm(info_list, total=len(info_list)):
            print(f"This is the length of sequence: {len(upstream + row['Alt_Sequence'] + downstream)}")
            encoded_seq_alt = one_hot_encode(upstream + row['Alt_Sequence'] + downstream)
            encoded_seq_ref = one_hot_encode(upstream + row['Ref_Sequence'] + downstream)
            
            # Create tensors
            rev_value = 0
            encoded_seq_with_rev_alt = [list(encoded_base) + [rev_value] for encoded_base in encoded_seq_alt]
            encoded_seq_with_rev_ref = [list(encoded_base) + [rev_value] for encoded_base in encoded_seq_ref]
            
            ref_tensor = torch.tensor(np.array(encoded_seq_with_rev_ref).reshape(1, SEQ_SIZE, 5).transpose(0, 2, 1), dtype=torch.float32).to(device)
            alt_tensor = torch.tensor(np.array(encoded_seq_with_rev_alt).reshape(1, SEQ_SIZE, 5).transpose(0, 2, 1), dtype=torch.float32).to(device)
            
            # Get model predictions
            pred_alt = model_rnn(alt_tensor)
            pred_ref = model_rnn(ref_tensor)
            
            pred_diff = (pred_ref.detach().cpu().flatten() - pred_alt.detach().cpu().flatten()).tolist()
            predictions.append(pred_diff)
        
        return predictions


    offsets = list(range(-90, 91, 1))

    predictions = {}

    k562 = pd.read_csv("/scratch/st-cdeboer-1/sambina/mpra/mpra_with_chromosome/agarwal/data_k562/fold_0/valid.txt", sep="\t")
    upstream = k562.iloc[0]['seq'][:15]
    downstream = k562.iloc[0]['seq'][-15:]


    for offset in offsets:
        info_list = [generate_sequences(row, sidemer_offset=offset) for index, row in snvs_only.iterrows()]
        pred_key = f"offset_{offset}"
        
        predictions[pred_key] = make_predictions(info_list, upstream, downstream)

    df = pd.DataFrame({key: value for key, value in predictions.items()})

    df.to_csv(output_path, index=False, compression="gzip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to generate predictions and save them to a CSV file.")
    
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model file.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output CSV file.")
    
    args = parser.parse_args()
    
    main(args.model_path, args.output_path)