#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rebuild `building_blocks_frag_mw250.csv` so that:

- Columns match `building_blocks_inland.csv`: smiles,id,size,price
- id/size/price are copied from inland when the SMILES correspond
  (matched via RDKit canonical SMILES).
- Rows in the frag file that do not exist in inland get synthetic IDs
  of the form FRAG_000001, FRAG_000002, ... (sizes/prices left as-is or empty).

Usage (from project root, in your conda env with RDKit installed):

    python scripts/rebuild_frag_metadata.py \
        --inland data/building_blocks_inland.csv \
        --frag data/building_blocks_frag_mw250.csv \
        --output data/building_blocks_frag_mw250.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Tuple, Optional

from rdkit import Chem  # type: ignore


def canon_smi(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None


def load_inland(path: str) -> Tuple[Dict[str, Dict[str, str]], set]:
    """
    Load `building_blocks_inland.csv` and build:
    - mapping: canonical_smiles -> {smiles,id,size,price}
    - existing_ids: set of all non-empty ids
    """
    mapping: Dict[str, Dict[str, str]] = {}
    existing_ids: set = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi_raw = (row.get("smiles") or "").strip()
            if not smi_raw:
                continue
            c = canon_smi(smi_raw)
            if c is None:
                continue
            if c not in mapping:
                rec = {
                    "smiles": smi_raw,
                    "id": (row.get("id") or "").strip(),
                    "size": (row.get("size") or "").strip(),
                    "price": (row.get("price") or "").strip(),
                }
                mapping[c] = rec
                if rec["id"]:
                    existing_ids.add(rec["id"])
    return mapping, existing_ids


def detect_smiles_column(fieldnames) -> str:
    """Best-effort detection of the SMILES column in a CSV."""
    if not fieldnames:
        raise ValueError("No columns found in frag CSV")
    lowered = [c.lower() for c in fieldnames]
    # Prefer exact 'smiles'
    for name, low in zip(fieldnames, lowered):
        if low == "smiles":
            return name
    # Then any column containing 'smile' / 'smiles'
    for name, low in zip(fieldnames, lowered):
        if "smiles" in low or "smile" in low:
            return name
    # Fallback: first column
    return fieldnames[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild fragment block metadata from inland library.")
    ap.add_argument("--inland", required=True, help="Path to building_blocks_inland.csv")
    ap.add_argument("--frag", required=True, help="Path to building_blocks_frag_mw250.csv (current)")
    ap.add_argument(
        "--output",
        required=True,
        help="Output CSV path for rebuilt frag library (can be same as --frag for in-place overwrite)",
    )
    args = ap.parse_args()

    if not os.path.exists(args.inland):
        raise SystemExit(f"inland CSV not found: {args.inland}")
    if not os.path.exists(args.frag):
        raise SystemExit(f"frag CSV not found: {args.frag}")

    inland_map, existing_ids = load_inland(args.inland)

    # Read frag CSV (1-col or multi-col)
    with open(args.frag, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise SystemExit("Frag CSV has no header / columns")
        col_smiles = detect_smiles_column(fieldnames)

        rows_out = []
        next_frag_idx = 1

        for row in reader:
            smi_raw = (row.get(col_smiles) or "").strip()
            if not smi_raw:
                continue
            c = canon_smi(smi_raw)
            rec_id = ""
            rec_size = ""
            rec_price = ""

            if c is not None and c in inland_map:
                src = inland_map[c]
                rec_id = src.get("id", "") or ""
                rec_size = src.get("size", "") or ""
                rec_price = src.get("price", "") or ""
            else:
                # Try to reuse any existing id/size/price columns from frag itself
                for key in ("id", "Id", "ID"):
                    if key in row and row.get(key):
                        rec_id = row[key].strip()
                        break
                for key in ("size", "Size"):
                    if key in row and row.get(key):
                        rec_size = row[key].strip()
                        break
                for key in ("price", "Price", "cost", "Cost"):
                    if key in row and row.get(key):
                        rec_price = row[key].strip()
                        break
                # If still no ID, allocate a unique FRAG_XXXXXX
                if not rec_id:
                    while True:
                        candidate = f"FRAG_{next_frag_idx:06d}"
                        next_frag_idx += 1
                        if candidate not in existing_ids:
                            rec_id = candidate
                            existing_ids.add(candidate)
                            break

            rows_out.append(
                {
                    "smiles": smi_raw,
                    "id": rec_id,
                    "size": rec_size,
                    "price": rec_price,
                }
            )

    # Write output
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["smiles", "id", "size", "price"])
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)


if __name__ == "__main__":
    main()


