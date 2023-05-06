import torch
from tqdm import tqdm
from Bio import SeqIO
model= torch.hub.load("facebookresearch/esm:main", 'esmfold_v1')
model = model.eval().cuda()

# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
model.set_chunk_size(20)

# Multimer prediction can be done with chains separated by ':'
file_path = 'data/DNAPred_Dataset/PDNA-41_sequence.fasta'
with torch.no_grad():
    for index, record in tqdm(SeqIO.parse(file_path, 'fasta')):
        output = model.infer_pdb(str(record.seq))
        with open(f"data/pred_pdb_files/{record.id}.pdb", "w") as f:
            f.write(output)
