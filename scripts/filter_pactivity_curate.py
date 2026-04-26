#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem.Scaffolds import MurckoScaffold
try:
    from tqdm import tqdm
    tqdm.pandas()
except Exception:
    tqdm = None


# ---------------------------
# RDKit helpers
# ---------------------------


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def calc_qed(smiles: str) -> Optional[float]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return float(QED.qed(mol))
    except Exception:
        return None


def _approximate_sa(mol: Chem.Mol) -> float:
    """Fallback synthetic accessibility approximation (1~10, lower is easier).

    This is used only if a proper SA_Score implementation isn't available.
    It's a heuristic combining size, ring systems, stereochemistry and macrocycles.
    """
    try:
        from rdkit.Chem import rdMolDescriptors as rdMD
    except Exception:
        rdMD = None

    num_heavy = mol.GetNumHeavyAtoms()
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings() if ring_info is not None else tuple()
    num_rings = len(atom_rings)
    has_macrocycle = any(len(r) >= 12 for r in atom_rings)
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)

    frac_sp3 = 0.5
    if rdMD is not None:
        try:
            frac_sp3 = float(rdMD.CalcFractionCSP3(mol))
        except Exception:
            frac_sp3 = 0.5

    # Base complexity grows with size
    score = 1.0 + 0.12 * max(0, num_heavy - 12)

    # Ring and macrocycles add difficulty
    score += 0.5 * max(0, num_rings - 1)
    if has_macrocycle:
        score += 1.0

    # Stereochemistry increases challenge
    score += 0.25 * len(chiral_centers)

    # Highly aromatic/flat systems are typically easier to make than crowded sp3
    # so penalize very high sp3 content slightly
    score += 0.8 * max(0.0, frac_sp3 - 0.6)

    # Clamp into 1..10
    return float(max(1.0, min(10.0, score)))


def calc_sa(smiles: str) -> Optional[float]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    # Try RDKit-contrib SA_Score if available
    try:
        from rdkit.Chem import SA_Score  # type: ignore

        try:
            # some distributions expose SA_Score.sascorer.calculateScore
            return float(SA_Score.sascorer.calculateScore(mol))  # type: ignore
        except Exception:
            pass
    except Exception:
        pass
    # Try standalone sascorer if installed
    try:
        import sascorer  # type: ignore

        return float(sascorer.calculateScore(mol))  # type: ignore
    except Exception:
        pass
    # Fallback approximation
    try:
        return _approximate_sa(mol)
    except Exception:
        return None


def murcko_scaffold_smiles(smiles: str) -> Optional[str]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(scaf, canonical=True)
    except Exception:
        return None


# ---------------------------
# Protein family classification
# ---------------------------


HYDROPHOBIC_AA: Set[str] = set(list("AVILMFWYC"))


def count_hydrophobic_helices(seq: str, window: int = 19, threshold: float = 0.68) -> int:
    """Very rough TM helix counter via hydrophobic windows.

    Not precise; only to enrich for 7TM-like proteins.
    """
    s = (seq or "").upper()
    if len(s) < window:
        return 0
    count = 0
    i = 0
    while i + window <= len(s):
        win = s[i : i + window]
        hyd = sum((1 if ch in HYDROPHOBIC_AA else 0) for ch in win)
        if hyd / float(window) >= threshold:
            count += 1
            i += window  # skip ahead to avoid counting overlapping windows for the same helix
        else:
            i += max(1, window // 3)
    return count


def is_kinase(seq: str) -> bool:
    if not isinstance(seq, str) or not seq:
        return False
    s = seq.upper()
    # Common protein kinase catalytic motifs
    motifs = [
        "HRDLKPEN",  # HRD motif
        "DLKPEN",    # shortened
        "DFG",       # DFG motif
        "VAIKVLK",   # conserved in many kinases
        "APE",       # activation segment
    ]
    return any(m in s for m in motifs)


def is_gpcr(seq: str) -> bool:
    if not isinstance(seq, str) or not seq:
        return False
    s = seq.upper()
    # Class A GPCR motifs (very approximate)
    has_dry = "DRY" in s
    # NPxxY motif: N P x x Y
    has_npxxy = False
    for i in range(0, len(s) - 4):
        if s[i] == "N" and s[i + 1] == "P" and s[i + 4] == "Y":
            has_npxxy = True
            break
    helices = count_hydrophobic_helices(s)
    # 7TM-like if >=6 hydrophobic helices and motif observed
    return (has_dry or has_npxxy) and helices >= 6


def classify_family(seq: str) -> Optional[str]:
    if is_kinase(seq):
        return "Kinase"
    if is_gpcr(seq):
        return "GPCR"
    return None


# ---------------------------
# Main curation pipeline
# ---------------------------


def curate(
    input_csv: str,
    output_csv: str,
    protein_filter: str = "gpcr_kinase",
    min_qed: float = 0.6,
    max_sa: float = 5.0,
    cap: int = 30000,
    seed: int = 42,
    family_map_csv: Optional[str] = None,
    scaffold_dedupe: bool = False,
    max_per_protein: int = 0,
    skip_sa: bool = False,
) -> None:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print(f"[Load] Reading: {input_csv}")
    df = pd.read_csv(input_csv)
    print(
        {
            "rows": len(df),
            "uniq_ligands": df.get("ligand_smiles", pd.Series(dtype=str)).nunique(),
            "uniq_proteins": df.get("protein_sequence", pd.Series(dtype=str)).nunique(),
        }
    )
    required = ["ligand_smiles", "protein_sequence"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Input CSV missing column '{c}'")

    # Build protein family map
    fam_map: Dict[str, str] = {}
    if family_map_csv and os.path.exists(family_map_csv):
        try:
            mdf = pd.read_csv(family_map_csv)
            cols = set(mdf.columns)
            key_col = "protein_sequence" if "protein_sequence" in cols else None
            fam_col = "family" if "family" in cols else None
            if key_col and fam_col:
                sub = mdf[[key_col, fam_col]].dropna()
                for _, row in sub.iterrows():
                    fam_map[str(row[key_col])] = str(row[fam_col])
            print({"family_map_loaded": len(fam_map)})
        except Exception:
            fam_map = {}

    # Classify proteins where missing from map
    def _label(seq: str) -> Optional[str]:
        if seq in fam_map:
            fam = fam_map[seq]
            if isinstance(fam, str) and fam:
                return fam
        return classify_family(seq)

    print("[Classify] Labeling protein families (GPCR/Kinase heuristics)...")
    if tqdm is not None:
        df["protein_family"] = df["protein_sequence"].astype(str).progress_apply(_label)
    else:
        df["protein_family"] = df["protein_sequence"].astype(str).apply(_label)

    # Apply protein filter
    pf = (protein_filter or "").lower()
    keep_fams: Set[str]
    if pf == "gpcr":
        keep_fams = {"GPCR"}
    elif pf == "kinase":
        keep_fams = {"Kinase"}
    else:
        keep_fams = {"GPCR", "Kinase"}

    before_pf = len(df)
    df = df[df["protein_family"].isin(list(keep_fams))].copy()
    after_pf = len(df)
    print({"family_filter": sorted(list(keep_fams)), "kept_rows": after_pf, "drop_rows": before_pf - after_pf})
    if df.empty:
        raise RuntimeError("No rows left after protein family filtering. Consider relaxing filters or providing a family map.")

    # Compute QED and SA per unique ligand
    unique_ligs: List[str] = (
        df["ligand_smiles"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().drop_duplicates().tolist()
    )

    qed_map: Dict[str, float] = {}
    sa_map: Dict[str, float] = {}
    # Score QED always
    print(f"[QED] Scoring {len(unique_ligs)} unique ligands (min_qed={min_qed})...")
    it_q = unique_ligs
    if tqdm is not None:
        it_q = tqdm(unique_ligs, desc="QED")
    for smi in it_q:
        q = calc_qed(smi)
        if q is not None:
            qed_map[smi] = q

    # Score SA unless skipped
    if not skip_sa:
        print(f"[SA] Scoring {len(unique_ligs)} unique ligands (max_sa={max_sa})...")
        it_s = unique_ligs
        if tqdm is not None:
            it_s = tqdm(unique_ligs, desc="SA")
        for smi in it_s:
            s = calc_sa(smi)
            if s is not None:
                sa_map[smi] = s

    def _passes(smi: str) -> bool:
        q = qed_map.get(smi, None)
        if q is None:
            return False
        if skip_sa:
            return q >= float(min_qed)
        s = sa_map.get(smi, None)
        # Be permissive if SA failed to compute; only filter when we have an SA value
        return (q >= float(min_qed)) and (s is None or s <= float(max_sa))

    lig_kept: List[str] = [s for s in unique_ligs if _passes(s)]
    print({"ligands_pass_filter": len(lig_kept), "skip_sa": bool(skip_sa)})
    if not lig_kept:
        if skip_sa:
            raise RuntimeError("No ligands pass QED threshold. Consider relaxing --min-qed.")
        raise RuntimeError("No ligands pass QED/SA thresholds. Consider relaxing --min-qed/--max-sa or add --skip-sa.")

    # Optional scaffold dedupe (one representative per scaffold)
    if scaffold_dedupe:
        seen_scaf: Set[str] = set()
        deduped: List[str] = []
        print("[Scaffold] Deduping by Murcko scaffold ...")
        it2 = lig_kept
        if tqdm is not None:
            it2 = tqdm(lig_kept, desc="Murcko")
        for smi in lig_kept:
            scaf = murcko_scaffold_smiles(smi) or f"NOSCAF:{smi}"
            if scaf in seen_scaf:
                continue
            seen_scaf.add(scaf)
            deduped.append(smi)
        lig_kept = deduped
        print({"after_scaffold_dedupe": len(lig_kept)})

    # Cap at N ligands (random, reproducible)
    if cap and cap > 0 and len(lig_kept) > cap:
        print({"apply_cap": int(cap), "before": len(lig_kept)})
        random.Random(int(seed)).shuffle(lig_kept)
        lig_kept = lig_kept[: int(cap)]
        print({"after_cap": len(lig_kept)})

    # Filter original rows by kept ligands
    df = df[df["ligand_smiles"].isin(lig_kept)].copy()

    # Optional per-protein cap to reduce over-representation
    if max_per_protein and max_per_protein > 0:
        print({"max_per_protein": int(max_per_protein)})
        df = (
            df.groupby("protein_sequence", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), int(max_per_protein)), random_state=int(seed)))
            .reset_index(drop=True)
        )

    # Attach qed/sa for transparency (not required by downstream scripts)
    df["qed"] = df["ligand_smiles"].map(qed_map)
    df["sa"] = df["ligand_smiles"].map(sa_map)

    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    print("[Save] Writing curated CSV ...")
    df.to_csv(output_csv, index=False)

    print(
        f"Curation complete. Families kept: {sorted(list(keep_fams))}. "
        f"Rows: {len(df)}; Unique ligands: {df['ligand_smiles'].nunique()} (cap={cap}).\n"
        f"Saved: {output_csv}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Filter protein_ligand_pactivity.csv to GPCR/Kinase and curate by QED/SA with a 30k cap.")
    p.add_argument("--input", default="data/protein_ligand_pactivity.csv", help="Input CSV path")
    p.add_argument("--output", default="data/protein_ligand_pactivity_curated.csv", help="Output CSV path")
    p.add_argument("--protein-filter", default="gpcr_kinase", choices=["gpcr", "kinase", "gpcr_kinase"], help="Protein family filter")
    p.add_argument("--min-qed", type=float, default=0.6, help="Minimum QED threshold")
    p.add_argument("--max-sa", type=float, default=5.0, help="Maximum SA threshold")
    p.add_argument("--cap", type=int, default=30000, help="Maximum number of unique ligands to keep")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--family-map", default=None, help="Optional CSV with columns protein_sequence,family to override heuristics")
    p.add_argument("--scaffold-dedupe", action="store_true", help="Keep at most one ligand per Murcko scaffold")
    p.add_argument("--max-per-protein", type=int, default=0, help="Optional cap per protein sequence (0=disabled)")
    p.add_argument("--skip-sa", action="store_true", help="Skip SA filtering; use QED-only")
    args = p.parse_args()

    curate(
        input_csv=args.input,
        output_csv=args.output,
        protein_filter=args.protein_filter,
        min_qed=args.min_qed,
        max_sa=args.max_sa,
        cap=args.cap,
        seed=args.seed,
        family_map_csv=args.family_map,
        scaffold_dedupe=bool(args.scaffold_dedupe),
        max_per_protein=int(args.max_per_protein),
        skip_sa=bool(args.skip_sa),
    )


if __name__ == "__main__":
    main()


