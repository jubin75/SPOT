#!/usr/bin/env python3
"""
Rank SMILES in an input .smi file by PLANTAIN score (lower is better),
exporting CSV with columns: smiles, plantain_min, plantain_reward.

Usage (CPU example):
  PYTHONPATH=/home/jb/phar python -u scripts/rank_plantain_from_smi.py \
    --smi /home/jb/phar/test/<PDBID>/compounds.smi \
    --pocket /home/jb/phar/test/<PDBID>/<PDBID>_pocket.pdb \
    --out /home/jb/phar/runs/plantain_ranked_from_smi.csv \
    --device cpu --scale 10 --limit -1

Notes:
  - This script loads PLANTAIN from lib/plantain and uses its pretrained weights.
  - Reward mapping uses the same exponential transform as training: exp(-score/scale) in (0,1].
"""

import argparse
import csv
import math
import os
import sys
import warnings
from typing import List, Dict

from Bio.PDB.PDBExceptions import PDBConstructionWarning  # ensure Bio installed


class _Chdir:
    def __init__(self, new_dir: str) -> None:
        self._new = new_dir
        self._old = os.getcwd()
    def __enter__(self):
        try:
            os.chdir(self._new)
        except Exception:
            pass
        return self
    def __exit__(self, exc_type, exc, tb):
        try:
            os.chdir(self._old)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank compounds.smi by PLANTAIN (outputs CSV)")
    ap.add_argument("--smi", required=True, help="Input .smi file (one SMILES per line)")
    ap.add_argument("--pocket", required=True, help="Pocket PDB (seed or expanded)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"], help="Device for PLANTAIN")
    ap.add_argument("--scale", type=float, default=10.0, help="Exponential reward scale (smaller => more sensitive)")
    ap.add_argument("--limit", type=int, default=-1, help="Only rank first N molecules; -1 for all")
    args = ap.parse_args()

    # Resolve paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    plantain_dir = os.path.join(project_root, "lib", "plantain")
    if plantain_dir not in sys.path:
        sys.path.insert(0, plantain_dir)

    # Import PLANTAIN modules after sys.path setup
    from common.cfg_utils import get_config
    from models.pretrained_plantain import get_pretrained_plantain
    from datasets.inference_dataset import InferenceDataset
    from terrace import collate

    # Read SMILES (robust to CSV header or multi-column files)
    def _read_smiles_maybe_csv(p: str) -> List[str]:
        import csv as _csv
        # Heuristic: if the first non-empty line contains a comma or tab, treat as CSV/TSV
        with open(p, "r", encoding="utf-8") as fh:
            lines = [ln.rstrip("\n\r") for ln in fh]
        nonempty = [ln for ln in lines if ln.strip()]
        if not nonempty:
            return []
        first = nonempty[0]
        is_table = ("," in first) or ("\t" in first)
        out: List[str] = []
        if is_table:
            # Try CSV first; fallback to TSV
            delim = "," if "," in first else "\t"
            reader = _csv.reader(nonempty, delimiter=delim)
            rows = list(reader)
            header = rows[0]
            # Detect smiles column
            def _idx(hdr: List[str]) -> int:
                cand = ["smiles", "SMILES", "ligand_smiles", "canonical_smiles"]
                lower = [h.strip().lower() for h in hdr]
                for c in cand:
                    if c.lower() in lower:
                        return lower.index(c.lower())
                return 0
            col = _idx(header)
            start = 1  # skip header
            for row in rows[start:]:
                if not row:
                    continue
                val = row[col].strip().strip('"').strip("'")
                if not val or val.lower() == "smiles":
                    continue
                out.append(val)
        else:
            for ln in nonempty:
                s = ln.strip().strip('"').strip("'")
                if not s or s.lower() == "smiles":
                    continue
                # If a stray comma/tab exists, take the first token as SMILES
                if "," in s:
                    s = s.split(",")[0].strip()
                if "\t" in s:
                    s = s.split("\t")[0].strip()
                out.append(s)
        # Validate SMILES to avoid RDKit parse errors downstream
        try:
            from rdkit import Chem as _Chem
            valid: List[str] = []
            for s in out:
                m = _Chem.MolFromSmiles(s)
                if m is not None:
                    valid.append(s)
            return valid
        except Exception:
            return out

    smiles_list: List[str] = _read_smiles_maybe_csv(args.smi)
    if args.limit and int(args.limit) > 0:
        smiles_list = smiles_list[: int(args.limit)]

    rows: List[Dict[str, str]] = []

    # Load PLANTAIN under its repo cwd to unlock relative paths (e.g., data/plantain_final.pt)
    with _Chdir(plantain_dir):
        cfg = get_config("icml", folder=os.path.join(plantain_dir, "configs"))
        model = get_pretrained_plantain()
        # Build a temporary dataset wrapper by writing a temp smi file
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            smi_tmp = os.path.join(td, "tmp.smi")
            with open(smi_tmp, "w", encoding="utf-8") as ff:
                for s in smiles_list:
                    ff.write(s + "\n")
            ds = InferenceDataset(cfg, smi_tmp, args.pocket, model.get_input_feats())

        # Device selection
        dev = str(args.device)
        if dev == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    dev = "cuda:0"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    dev = "mps"
                else:
                    dev = "cpu"
            except Exception:
                dev = "cpu"
        try:
            model = model.to(dev)
        except Exception:
            dev = "cpu"
            model = model.to("cpu")
        model.eval()

        # Rank
        for i in range(len(ds)):
            x, y = ds[i]
            smi = smiles_list[i]
            if not getattr(y, "success", True):
                rows.append({"smiles": smi, "plantain_min": "", "plantain_reward": ""})
                continue
            batch = collate([x])
            try:
                batch = batch.to(dev)
            except Exception:
                pass
            try:
                pred = model(batch)[0]
                s0 = float(pred.score[0].detach().cpu().item()) if hasattr(pred, "score") else None
            except Exception:
                s0 = None
            r0 = None if s0 is None else math.exp(-float(s0) / max(1e-6, float(args.scale)))
            # clamp reward to [0,1]
            if r0 is not None:
                r0 = float(min(1.0, max(0.0, r0)))
            rows.append({
                "smiles": smi,
                "plantain_min": "" if s0 is None else f"{s0:.6f}",
                "plantain_reward": "" if r0 is None else f"{r0:.6f}",
            })

    # Sort by reward desc
    rows.sort(key=lambda d: float(d.get("plantain_reward") or 0.0), reverse=True)

    # Save CSV
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["smiles", "plantain_min", "plantain_reward"])
        w.writeheader(); w.writerows(rows)
    print({"saved": args.out, "rows": len(rows)})


if __name__ == "__main__":
    # Silence PDB construction warnings to reduce noise
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        main()


