import os

from Bio.PDB.PDBList import PDBList


def download_pdb_files(pdb_ids, save_dir="data/raw"):
    """Download associated PDB files for a given list of PDB IDs"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Deals with duplicates because download_pdb_files downloads .ent and won't detect duplicate .pdb. Not necessarily necessary.
    for pdb_id in pdb_ids:
        pdb_file = os.path.join(save_dir, f"{pdb_id.lower()}.pdb")
        if os.path.exists(pdb_file):
            pdb_ids.remove(pdb_id)

    # Actual Downloading of files
    pdbl = PDBList()
    pdbl.download_pdb_files(pdb_ids, pdir=save_dir, file_format="pdb")

    # Rename .ent files to .pdb for compatibility
    for pdb_id in pdb_ids:
        ent_file = os.path.join(save_dir, f"pdb{pdb_id.lower()}.ent")
        pdb_file = os.path.join(save_dir, f"{pdb_id.lower()}.pdb")
        if os.path.exists(ent_file):
            os.rename(ent_file, pdb_file)


if __name__ == "__main__":
    file = open("data/pdb_list.txt", "r")
    pdb_ids = [id[0:4] for id in file.readlines()]
    file.close()
    download_pdb_files(pdb_ids)
