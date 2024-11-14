import os

import Bio.PDB.PDBParser as PDBParser
import numpy as np


def calculate_dist_matrix(name, pdb_file):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(name, pdb_file)

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
    name = "ubiquitin"
    pdb_file = "./data/raw/1ubq.pdb"
    path = "./data/raw/"

    files = [
        os.path.join(path, file) for file in os.listdir(path) if file.endswith(".pdb")
    ]
    for file in files:
        name = file[-8:-4]

        matrix = calculate_dist_matrix(name, file)

        matrix_filename = os.path.join(
            "./data/processed", f"{name}_distance_matrix.csv"
        )
        os.makedirs(os.path.dirname(matrix_filename), exist_ok=True)

        np.savetxt(matrix_filename, matrix, delimiter=",")
        print(f"Saved distance matrix for {name} to {matrix_filename}")
