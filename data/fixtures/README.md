# Fixtures

Six real PDB files (~500 KB total), copied unmodified from RCSB, so the pipeline
can be exercised and tested without downloading the full 800-entry set.

| entry | why it is here |
|-------|----------------|
| `1ae4.pdb` | long chain, yields 10 windows |
| `1uha.pdb` | short usable chain, yields 2 windows |
| `2acy.pdb` | yields 3 windows |
| `1acf.pdb` | hashes into the validation split |
| `1j3a.pdb` | hashes into the validation split |
| `1aie.pdb` | **yields nothing**: no contiguous run of 64 residues, so it exercises the skip path |

Try the pipeline on them:

```bash
folduzz preprocess --raw data/fixtures/raw --out /tmp/folduzz-fixture
```

Provenance: `https://files.rcsb.org/download/<ID>.pdb`, fetched 2026-10-01.
These files are PDB archive entries and carry the RCSB PDB terms of use
(public domain / CC0 for the coordinate data).
