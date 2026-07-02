from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor
from alphagenome_pytorch.named_outputs import TrackMetadataCatalog
import pyfaidx
import numpy as np

model = AlphaGenome.from_pretrained(
    "/scratch/st-cdeboer-1/sambina/position_mpra/outputs/8-aphagenome/pytorch/model_all_folds.safetensors"
)
catalog = TrackMetadataCatalog.load_builtin(organism=0)  # 0 = human
model.set_track_metadata_catalog(catalog)


with pyfaidx.Fasta("/scratch/st-cdeboer-1/sambina/reference_genome/hg38.fa") as genome:
    sequence = str(genome["chr1"][1_000_000:1_131_072])

dna_onehot = sequence_to_onehot_tensor(sequence).unsqueeze(0)

preds = model.predict(dna_onehot, organism_index=0, named_outputs=True)  # 0=human, 1=mouse


rna128 = preds.rna_seq[128]
hepg2 = rna128.select(biosample_name="HepG2")
print("HepG2 tracks:", hepg2.num_tracks, "shape:", hepg2.shape)
hepg2_avg = hepg2.tensor.cpu().numpy()
hepg2_avg = hepg2_avg.mean(axis=-1)

print("HepG2 averaged shape:", hepg2_avg.shape)
print(hepg2_avg)

# Save it
np.save(
    "/scratch/st-cdeboer-1/sambina/position_mpra/src/8-predict_alphagenome/hepg2_avg.npy", 
    hepg2_avg
)
print("saved hepg2_avg.npy with shape", hepg2_avg.shape)