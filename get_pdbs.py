import os

from Bio.PDB.PDBList import PDBList


def download_pdb_files(pdb_ids, save_dir="data/raw"):
    """Download associated PDB files for a given list of PDB IDs"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pdbl = PDBList()
    for id in pdb_ids:
        pdbl.retrieve_pdb_file(id, pdir=save_dir, file_format="pdb")


if __name__ == "__main__":
    file = open("data/pdb_list.txt", "r")
    pdb_ids = [id[0:4] for id in file.readlines()]
    file.close()
    download_pdb_files(pdb_ids)
