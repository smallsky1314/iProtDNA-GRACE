import numpy as np
import csv
from Bio import SeqIO
from Bio import PDB
import warnings

from Bio.PDB.PDBExceptions import PDBConstructionWarning

three_to_one = {'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'GLU': 'E', 'PCA': 'E', 'GLN': 'Q',
                'ASP': 'D', 'ASN': 'N', 'HIS': 'H', 'TRP': 'W', 'PHE': 'F', 'TYR': 'Y',
                'PTR': 'Y', 'ARG': 'R', 'LYS': 'K', 'SER': 'S', 'THR': 'T', 'MET': 'M',
                'MSE': 'M', 'ALA': 'A', 'GLY': 'G', 'PRO': 'P', 'CYS': 'C', 'OCS': 'C',
                'CME': 'C', 'SMC': 'C', 'CSO': 'C'}


def distance_matrix(pdb_id, pdb_path, chain_id, seq):
    print(pdb_id + chain_id)
    parser = PDB.PDBParser()

    structure = parser.get_structure(pdb_id, pdb_path)

    i = 0
    ca_coords = []  # [n , (array(3, )]
    chain = structure[0][chain_id]

    for residue in chain:
        if "CA" in residue and len(residue.get_resname()) == 3 and i < len(seq):
            if three_to_one[residue.get_resname()] == seq[i]:
                ca_atom = residue["CA"]
                ca_coord = ca_atom.get_coord()
                ca_coords.append(ca_coord)
                i += 1
    if len(ca_coords) == len(seq):
        ca_coords = np.array(ca_coords)

        dist_matrix = np.zeros((len(ca_coords), len(ca_coords)))
        for i, coord1 in enumerate(ca_coords):
            for j, coord2 in enumerate(ca_coords):
                dist = np.sqrt(np.sum((coord1 - coord2) ** 2))
                dist_matrix[i, j] = dist

        with open('./data/distance_matrix_335_52/' + pdb_id + chain_id + '.csv', mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for row in dist_matrix:
                writer.writerow(row)
    else:
        print('ERROR: ' + pdb_id + chain_id)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=PDBConstructionWarning)
    file_path = '../data/PDNA-52_sequence.fasta'
    for index, record in enumerate(SeqIO.parse(file_path, 'fasta')):
        pdb_path = './data/pdb_52/' + record.id[0:4] + '.pdb'
        distance_matrix(record.id[0:4], pdb_path, record.id[4], str(record.seq))
