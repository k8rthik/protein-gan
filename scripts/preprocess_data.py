import os

import numpy as np
from Bio.PDB.PDBParser import PDBParser


def calculate_dist_matrix(pdb_file):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_file[-8:-4], pdb_file)

    alpha_carbons = []
    for model in structure.get_models():
        for chain in model.get_chains():
            for residue in chain.get_residues():
                if "CA" in residue:
                    alpha_carbons.append(residue["CA"].coord)

    ca_coords = np.array(alpha_carbons)
    distance_matrix = np.sqrt(
        ((ca_coords[:, None, :] - ca_coords[None, :, :]) ** 2).sum(-1)
    )

    return distance_matrix


if __name__ == "__main__":
    path = "./data/raw/"
    matrices = {
        file[0:4]: calculate_dist_matrix(os.path.join(path, file))
        for file in os.listdir(path)
    }

    for matrix in matrices:
        print(matrices[matrix])
        np.savetxt("data/processed/" + matrix + ".csv", matrices[matrix], delimiter=",")
