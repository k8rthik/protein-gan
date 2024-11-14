import os

import numpy as np
from Bio.PDB.PDBParser import PDBParser


def calculate_dist_matrix(pdb_file):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    alpha_carbons = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    alpha_carbons.append(residue["CA"].coord)

    ca_coords = np.array(alpha_carbons)
    distance_matrix = np.sqrt(
        ((ca_coords[:, None, :] - ca_coords[None, :, :]) ** 2).sum(-1)
    )

    return distance_matrix


if __name__ == "__main__":
    path = "./data/raw/"
    files = [os.path.join(path, pdb) for pdb in os.listdir(path)]
    for file in files:
        np.savetxt(
            "data/processed/" + file[-8:-4] + ".csv",
            calculate_dist_matrix(file),
            delimiter=",",
        )
