#!/usr/bin/env python3
"""
Expand a receptor pocket region by selecting residues within a given radius
around the centroid of a seed pocket PDB.

Usage:
  python expand_pocket.py receptor.pdb pocket_seed.pdb out_pocket.pdb --radius 12

Notes:
  - The script keeps only standard amino-acid residues (excludes HETATM/water).
  - The radius can be set between 10–12 Å typically; default is 12 Å.
  - The output PDB will contain a single model with only selected residues.
"""

import argparse
import os
import warnings
import copy
from typing import Tuple

import numpy as np
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import Structure, Model, Chain


def load_structure(pdb_path: str, struct_id: str):
    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        return parser.get_structure(struct_id, pdb_path)


def get_centroid(struct) -> np.ndarray:
    coords = []
    for atom in struct.get_atoms():
        try:
            coords.append(atom.get_coord())
        except Exception:
            pass
    if not coords:
        raise ValueError("No atoms found in pocket seed PDB")
    arr = np.vstack(coords)
    return arr.mean(axis=0)


def residue_within_radius(residue, center: np.ndarray, radius: float) -> bool:
    for atom in residue.get_atoms():
        try:
            if float(np.linalg.norm(atom.get_coord() - center)) <= radius:
                return True
        except Exception:
            continue
    return False


def build_expanded_pocket(receptor_pdb: str, pocket_seed_pdb: str, out_pocket_pdb: str, radius: float) -> Tuple[str, float]:
    rec = load_structure(receptor_pdb, "REC")
    poc = load_structure(pocket_seed_pdb, "POC")
    center = get_centroid(poc)

    model = next(rec.get_models())
    new_struct = Structure.Structure("POCKET")
    new_model = Model.Model(0)
    new_struct.add(new_model)

    for chain in model:
        new_chain = Chain.Chain(chain.id)
        kept_any = False
        for res in chain:
            # Keep only standard amino acids (exclude HETATM, water with hetflag != ' ')
            hetflag = res.id[0]
            if str(hetflag).strip() != '':
                continue
            if residue_within_radius(res, center, radius):
                new_chain.add(copy.deepcopy(res))
                kept_any = True
        if kept_any:
            new_model.add(new_chain)

    io = PDBIO()
    io.set_structure(new_struct)
    os.makedirs(os.path.dirname(out_pocket_pdb) or ".", exist_ok=True)
    io.save(out_pocket_pdb)
    return out_pocket_pdb, radius


def main() -> None:
    ap = argparse.ArgumentParser(description="Expand receptor pocket around seed centroid with a given radius (Å)")
    ap.add_argument("receptor_pdb", type=str, help="Receptor PDB path")
    ap.add_argument("pocket_seed_pdb", type=str, help="Seed pocket PDB path (defines centroid)")
    ap.add_argument("out_pocket_pdb", type=str, help="Output expanded pocket PDB path")
    ap.add_argument("--radius", type=float, default=12.0, help="Selection radius in Å (default: 12.0)")
    args = ap.parse_args()

    out_path, rad = build_expanded_pocket(args.receptor_pdb, args.pocket_seed_pdb, args.out_pocket_pdb, float(args.radius))
    print({"saved": out_path, "radius": rad})


if __name__ == "__main__":
    main()


