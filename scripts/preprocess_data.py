import os
from inspect import currentframe

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


def standardize_matrix(matrix, target_size=64):
    current_size = matrix.shape[0]
    if current_size < target_size:
        padded_matrix = np.zeros((target_size, target_size))
        padded_matrix[:current_size, :current_size] = matrix
        return padded_matrix
    elif current_size > target_size:
        return matrix[:target_size, :target_size]
    else:
        return matrix


def normalize_matrix(matrix):
    min_val, max_val = matrix.min(), matrix.max()
    normalized = (matrix - min_val) / (max_val - min_val) * 2 - 1
    return normalized


if __name__ == "__main__":
    path = "./data/raw/"
    processed_path = "./data/processed/"
    target_size = 64

    os.makedirs(processed_path, exist_ok=True)

    for file in os.listdir(path):
        if file.endswith(".pdb"):
            pdb_file = os.path.join(path, file)
            matrix_id = file[0:4]
            distance_matrix = calculate_dist_matrix(matrix_id, pdb_file)

            standardized_matrix = standardize_matrix(
                distance_matrix, target_size=target_size
            )
            normalized_matrix = normalize_matrix(standardized_matrix)

            output_file = os.path.join(processed_path, f"{matrix_id}.csv")
            np.savetxt(output_file, normalized_matrix, delimiter=",")

            print(f"Processed and save {output_file}")

    # files = [
    #     os.path.join(path, file) for file in os.listdir(path) if file.endswith(".pdb")
    # ]
    # for file in files:
    #     name = file[-8:-4]
    #
    #     matrix = calculate_dist_matrix(name, file)
    #
    #     matrix_filename = os.path.join(
    #         "./data/processed", f"{name}_distance_matrix.csv"
    #     )
    #     os.makedirs(os.path.dirname(matrix_filename), exist_ok=True)
    #
    #     np.savetxt(matrix_filename, matrix, delimiter=",")
    #     print(f"Saved distance matrix for {name} to {matrix_filename}")
