#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from typing import List

def main() -> None:
    ap = argparse.ArgumentParser(description="Filter building blocks by molecular weight and deduplicate SMILES")
    ap.add_argument("--input", required=True, help="Input CSV with a SMILES column (name flexible)")
    ap.add_argument("--output", required=True, help="Output CSV path for filtered blocks")
    ap.add_argument("--max-mw", type=float, default=250.0, help="Keep blocks with MolWt <= max-mw (Da)")
    ap.add_argument("--smiles-col", type=str, default="smiles", help="SMILES column name; if absent, use first column")
    args = ap.parse_args()

    import pandas as pd  # type: ignore
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import Descriptors  # type: ignore
    from rdkit import RDLogger  # type: ignore

    # Silence RDKit parse errors
    try:
        RDLogger.DisableLog('rdApp.error')
    except Exception:
        pass

    df = pd.read_csv(args.input, low_memory=False)
    col = args.smiles_col if args.smiles_col in df.columns else df.columns[0]
    # Drop NaN before converting to string to avoid literal 'nan' tokens
    series = df[col]
    series = series[~series.isna()]
    series = series.astype(str).str.strip()
    # Remove empty and NA-like placeholders and template tokens
    na_like = {"nan", "na", "n/a", "none", "null"}
    def _is_valid_token(s: str) -> bool:
        if not s:
            return False
        ls = s.lower()
        if ls in na_like:
            return False
        if ('<' in s) or ('>' in s):
            return False
        return True
    series = series[series.map(_is_valid_token)]

    seen = set()
    kept: List[str] = []
    thr = float(args.max_mw)
    for s in series.tolist():
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            mw = float(Descriptors.MolWt(m))
            if mw <= thr:
                can = Chem.MolToSmiles(m)
                if can and can not in seen:
                    seen.add(can)
                    kept.append(can)
        except Exception:
            continue

    out_df = pd.DataFrame({"smiles": kept})
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print({"input": args.input, "kept": len(kept), "max_mw": thr, "output": args.output})

if __name__ == "__main__":
    main()


