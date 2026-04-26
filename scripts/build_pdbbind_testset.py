#!/usr/bin/env python
from __future__ import annotations

"""
Build a PDBBind refined set-based test set by excluding proteins present in training.

Inputs:
- data/PdbBind_v2020_refined/<PDB_ID>/*_protein.pdb (SEQRES records)
- data/protein_ligand_pactivity_curated.csv (fallback: data/protein_ligand_pactivity.csv)

Outputs:
- data/pdbbind_testset.csv with columns:
  [pdb_id, protein_sequence, ligand_smiles_list, num_ligands]
- data/pdbbind_testset.fasta

Logic:
- Extract one-letter FASTA from SEQRES (3-letter codes) of *_protein.pdb
- Parse ligand molecules from .mol2 and .sdf under each PDB folder and convert to SMILES
- Exclude proteins whose sequence exactly matches or is a substring/superset of any training protein sequence
"""

import argparse
import os
import glob
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolToSmiles


AA3_TO_1: Dict[str, str] = {
    # Standard amino acids
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common ambiguous or modified placeholders (map to X)
    "ASX": "B", "GLX": "Z", "SEC": "U", "PYL": "O",
}


def _sanitize_seq(seq: str) -> str:
    s = (seq or "").upper()
    return "".join(ch for ch in s if ch.isalpha())


def parse_seqres_to_fasta(pdb_path: str) -> str:
    """Extract a one-letter sequence from SEQRES records of a PDB file.

    Concatenates residues across chains in file order. Non-mapped residues -> 'X'.
    """
    if not os.path.exists(pdb_path):
        return ""
    residues: List[str] = []
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("SEQRES"):
                    continue
                # Residue names typically begin at col 20; separated by spaces
                parts = line[19:].strip().split()
                # Filter out chain index and residue count if present
                # SEQRES format: cols [12-14]=serNum, [12]=?; simpler: take last up to 13 tokens
                for tok in parts:
                    aa3 = tok.strip().upper()
                    if len(aa3) < 3:
                        # Likely chain/serial numbers; skip
                        continue
                    one = AA3_TO_1.get(aa3, None)
                    residues.append(one if one is not None else "X")
    except Exception:
        return ""
    return _sanitize_seq("".join(residues))


def ligand_smiles_in_folder(folder: str) -> List[str]:
    smiles: List[str] = []
    # Prefer SDF, then MOL2
    sdf_paths = glob.glob(os.path.join(folder, "*.sdf")) + glob.glob(os.path.join(folder, "*ligand*.sdf"))
    mol2_paths = glob.glob(os.path.join(folder, "*.mol2")) + glob.glob(os.path.join(folder, "*ligand*.mol2"))

    # SDF: iterate suppliers
    for p in sorted(set(sdf_paths)):
        try:
            suppl = Chem.SDMolSupplier(p, removeHs=False)
            for m in suppl:
                if m is None:
                    continue
                s = MolToSmiles(m, isomericSmiles=True)
                if s:
                    smiles.append(s)
        except Exception:
            continue
    # MOL2: one molecule per file typically
    for p in sorted(set(mol2_paths)):
        try:
            m = Chem.MolFromMol2File(p, sanitize=True, removeHs=False)
            if m is None:
                continue
            s = MolToSmiles(m, isomericSmiles=True)
            if s:
                smiles.append(s)
        except Exception:
            continue
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for s in smiles:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def load_training_proteins(train_csv: str) -> List[str]:
    df = pd.read_csv(train_csv)
    if "protein_sequence" not in df.columns:
        return []
    seqs = (
        df["protein_sequence"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().drop_duplicates().tolist()
    )
    return [_sanitize_seq(s) for s in seqs]


def is_subset_or_equal(a: str, b: str) -> bool:
    """Return True if a is substring of b or b is substring of a."""
    if not a or not b:
        return False
    return (a in b) or (b in a)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build PDBBind refined test set by excluding training proteins")
    p.add_argument("--pdbbind", default="data/PdbBind_v2020_refined", help="Path to PDBBind refined root")
    p.add_argument("--train-csv", default="data/protein_ligand_pactivity_curated.csv", help="Training CSV with protein_sequence column (fallback to protein_ligand_pactivity.csv if not found)")
    p.add_argument("--output-csv", default="data/pdbbind_testset.csv")
    p.add_argument("--output-fasta", default="data/pdbbind_testset.fasta")
    p.add_argument("--min-seq-len", type=int, default=30, help="Minimum protein sequence length to keep")
    p.add_argument("--max-ligands-per-entry", type=int, default=8, help="Cap number of ligands saved per PDB entry")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    pdb_root = args.pdbbind
    if not os.path.isdir(pdb_root):
        raise FileNotFoundError(f"PDBBind refined directory not found: {pdb_root}")

    train_csv = args.train_csv
    if not os.path.exists(train_csv):
        alt = os.path.join(os.path.dirname(train_csv) or "data", "protein_ligand_pactivity.csv")
        train_csv = alt if os.path.exists(alt) else args.train_csv

    train_seqs = load_training_proteins(train_csv)
    train_set = set(train_seqs)
    print({"training_unique_proteins": len(train_set), "train_csv": train_csv})

    rows: List[Dict[str, str]] = []
    subdirs = [d for d in sorted(os.listdir(pdb_root)) if os.path.isdir(os.path.join(pdb_root, d))]
    print({"pdbbind_entries": len(subdirs)})

    kept = 0
    for i, pdbid in enumerate(subdirs, start=1):
        folder = os.path.join(pdb_root, pdbid)
        # Find protein PDB file
        pdb_candidates = glob.glob(os.path.join(folder, "*_protein.pdb")) + glob.glob(os.path.join(folder, "*protein*.pdb"))
        if not pdb_candidates:
            continue
        pdb_file = sorted(pdb_candidates)[0]
        seq = parse_seqres_to_fasta(pdb_file)
        seq = _sanitize_seq(seq)
        if not seq or len(seq) < int(args.min_seq_len):
            continue
        # Exclude if matches/contains a training sequence or vice versa
        exclude = False
        if seq in train_set:
            exclude = True
        else:
            # substring check against a sample of training set to avoid O(N^2). Use full set if small.
            # For robustness, scan all since typical train set is ~30k
            for t in train_set:
                if is_subset_or_equal(seq, t):
                    exclude = True
                    break
        if exclude:
            continue

        ligs = ligand_smiles_in_folder(folder)
        if not ligs:
            continue
        capped = ligs[: int(args.max_ligands_per_entry)]
        rows.append({
            "pdb_id": pdbid,
            "protein_sequence": seq,
            "ligand_smiles_list": " | ".join(capped),
            "num_ligands": str(len(capped)),
        })
        kept += 1
        if i % 200 == 0:
            print({"progress": i, "kept": kept})

    if not rows:
        print({"warning": "No PDBBind test entries kept after filtering"})

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_csv) or "data", exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    # Write FASTA
    with open(args.output_fasta, "w", encoding="utf-8") as f:
        for _, r in out_df.iterrows():
            f.write(f">{r['pdb_id']}\n")
            f.write(str(r["protein_sequence"]) + "\n")
    print({"saved_csv": args.output_csv, "saved_fasta": args.output_fasta, "entries": len(out_df)})


if __name__ == "__main__":
    main()


