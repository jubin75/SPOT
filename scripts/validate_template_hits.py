from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd
from rdkit import Chem
from LeadGFlowNet.template_expander import TemplateLibrary
from rdkit import RDLogger

# Silence RDKit warnings/errors during batch validation
RDLogger.DisableLog('rdApp.warning')
try:
    RDLogger.DisableLog('rdApp.error')
except Exception:
    pass


def load_blocks(csv_path: str, cap: int = 5000, max_mw: float = 200.0) -> List[str]:
    df = pd.read_csv(csv_path, usecols=["smiles"])  # expect 'smiles'
    col = df["smiles"].dropna().astype(str).str.strip()
    from rdkit.Chem import Descriptors
    seen = set(); kept: List[str] = []
    for s in col.tolist():
        if not s or s in seen:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        mw = Descriptors.MolWt(m)
        if mw <= max_mw:
            can = Chem.MolToSmiles(m)
            if can not in seen:
                seen.add(can)
                kept.append(can)
        if len(kept) >= cap:
            break
    return kept


def main() -> None:
    ap = argparse.ArgumentParser("Validate template hits against external blocks")
    ap.add_argument("--template-csv", type=str, default="data/top100/template_top100.xlsx")
    ap.add_argument("--blocks-csv", type=str, default="data/building_blocks_inland.csv")
    ap.add_argument("--states", type=str, default="data/forward_trajectories.csv", help="CSV with column state_smiles to seed states")
    ap.add_argument("--max-states", type=int, default=200)
    ap.add_argument("--try-templates", type=int, default=128)
    ap.add_argument("--sample-blocks", type=int, default=256)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.template_csv):
        raise FileNotFoundError(args.template_csv)
    if not os.path.exists(args.blocks_csv):
        raise FileNotFoundError(args.blocks_csv)
    if not os.path.exists(args.states):
        raise FileNotFoundError(args.states)

    lib = TemplateLibrary.from_csv(args.template_csv, max_rows=int(args.max_rows))
    print({"templates_loaded": len(getattr(lib, "compiled", [])), "src": args.template_csv})

    blocks = load_blocks(args.blocks_csv, cap=5000, max_mw=200.0)
    print({"blocks_loaded": len(blocks), "src": args.blocks_csv})

    sdf = pd.read_csv(args.states, usecols=["state_smiles"]).dropna().drop_duplicates().head(int(args.max_states))
    states = sdf["state_smiles"].astype(str).tolist()
    total = 0
    hit_states = 0
    hit_products = 0
    for st in states:
        total += 1
        prods = lib.propose_products_with_pool(
            state_smiles=st,
            block_pool=blocks,
            try_limit_templates=int(args.try_templates),
            sample_blocks=int(args.sample_blocks),
            max_products_per_template=1,
        )
        if prods:
            hit_states += 1
            hit_products += len(prods)
    print({
        "states": total,
        "hit_states": hit_states,
        "hit_rate": (hit_states / max(1, total)),
        "products_found": hit_products,
        "avg_products_per_hit_state": (hit_products / max(1, hit_states)) if hit_states else 0.0,
    })


if __name__ == "__main__":
    main()


