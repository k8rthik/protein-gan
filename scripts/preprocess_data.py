import numpy as np
from Bio.PDB.PDBParser import PDBParser


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

    distance_matrix = calculate_dist_matrix(name, pdb_file)

    np.savetxt("data/processed/distance_matrix.csv", distance_matrix, delimiter=",")
    print("Pairwise distance matrix: ")
    print(distance_matrix)
