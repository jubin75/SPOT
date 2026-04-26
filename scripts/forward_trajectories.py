"""
Forward Trajectories Builder

Convert retrosynthesis dataset (reaction_paths_all_routes.csv) into
forward one-step synthesis trajectories suitable for training.

Example conversion:
  B_final -> {B_intermediate, C}
  B_intermediate -> {D, E}
Produces two training samples:
  (state: D, action: (choose building block E, choose reaction R1)) -> result: B_intermediate
  (state: B_intermediate, action: (choose building block C, choose reaction R2)) -> result: B_final

CLI:
  python scripts/forward_trajectories.py \
      --input data/reaction_paths_all_routes.csv \
      --output data/forward_trajectories.csv \
      [--skip-start-steps]

Output columns:
- ligand_smiles
- route_id
- step_index
- route_score
- state_smiles
- action_building_block
- action_reaction_template
- result_smiles
- is_start_state
- num_intermediate_candidates
- num_building_blocks
"""

from __future__ import annotations

import argparse
import math
from typing import List, Dict, Tuple

import pandas as pd
from SynthPolicyNet.data_utils import mol_from_smiles


# Source CSV columns (Chinese headers)
COL_LIGAND = "ligand_smiles"
COL_ROUTE_ID = "解析树ID"
COL_STEP_INDEX = "步骤序号"
COL_ROUTE_SCORE = "路线分数"
COL_PRODUCT = "当前状态分子"
COL_STOCK_REACTANTS = "在zinc库里的反应物砌块分子"
COL_INTERMEDIATES = "和反应物砌块分子反应的中间体分子"
COL_TEMPLATE = "反应模版"


def is_na_like(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return True
        if s.lower() in {"n/a", "na", "nan"}:
            return True
    return False


def _split_on_plus_outside_brackets(s: str) -> List[str]:
    """Split on top-level ' + ' separators, not on '+' inside SMILES brackets.

    Heuristic: treat a '+' as a separator only when it has whitespace on both sides
    and we are not inside '[' ... ']'. This avoids splitting charged atoms like
    '[N+](C)(C)C'.
    """
    if not s:
        return []
    parts: List[str] = []
    buf: List[str] = []
    bracket_depth = 0
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if ch == '[':
            bracket_depth += 1
        elif ch == ']':
            bracket_depth = max(0, bracket_depth - 1)

        if ch == '+' and bracket_depth == 0:
            prev_c = s[i - 1] if i - 1 >= 0 else ''
            next_c = s[i + 1] if i + 1 < n else ''
            if prev_c.isspace() and next_c.isspace():
                token = ''.join(buf).strip()
                if token:
                    parts.append(token)
                buf = []
                i += 1
                continue

        buf.append(ch)
        i += 1

    token = ''.join(buf).strip()
    if token:
        parts.append(token)
    return parts


def parse_plus_list(value: object) -> List[str]:
    if is_na_like(value):
        return []
    if not isinstance(value, str):
        return []
    # Prefer splitting on ' + ' outside of brackets; fallback to single token
    items = _split_on_plus_outside_brackets(value)
    if not items:
        # As a last resort, try splitting by ' + ' literal
        if ' + ' in value:
            items = [item.strip() for item in value.split(' + ')]
        else:
            items = [value.strip()]
    return [x for x in items if x]


def _is_valid_smiles(smiles: str) -> bool:
    try:
        return bool(mol_from_smiles(smiles))
    except Exception:
        return False


def _safe_str(value: object) -> str:
    """Convert to string while mapping NA-like values to empty string."""
    return "" if is_na_like(value) else str(value)


def _mol_weight(smiles: str) -> float:
    """Compute molecular weight using RDKit; return +inf on failure."""
    try:
        mol = mol_from_smiles(smiles)
        if not mol:
            return float("inf")
        from rdkit.Chem import Descriptors  # type: ignore
        return float(Descriptors.MolWt(mol))
    except Exception:
        return float("inf")


def load_dataset(path: str) -> pd.DataFrame:
    usecols = [
        COL_LIGAND,
        COL_ROUTE_ID,
        COL_STEP_INDEX,
        COL_ROUTE_SCORE,
        COL_PRODUCT,
        COL_STOCK_REACTANTS,
        COL_INTERMEDIATES,
        COL_TEMPLATE,
    ]
    df = pd.read_csv(path, usecols=usecols)
    df[COL_STEP_INDEX] = pd.to_numeric(df[COL_STEP_INDEX], errors="coerce").astype("Int64")
    return df


def convert_row(
    row: pd.Series,
    skip_start_steps: bool,
    max_block_mw: float | None,
    max_state_mw: float | None = None,
    max_ligand_mw: float | None = None,
) -> List[dict]:
    product = _safe_str(row.get(COL_PRODUCT, ""))
    if product and not _is_valid_smiles(product):
        # Skip rows where the product SMILES is invalid
        return []
    # Ligand MW filter (if provided)
    ligand = _safe_str(row.get(COL_LIGAND, ""))
    if ligand and max_ligand_mw is not None:
        try:
            thr_lig = float(max_ligand_mw)
        except Exception:
            thr_lig = None
        if thr_lig is not None and thr_lig > 0:
            if _mol_weight(ligand) > thr_lig:
                return []
    blocks_all = [b for b in parse_plus_list(row.get(COL_STOCK_REACTANTS)) if _is_valid_smiles(b)]
    if max_block_mw is not None:
        try:
            thr = float(max_block_mw)
        except Exception:
            thr = None
        if thr is not None and thr > 0:
            blocks = [b for b in blocks_all if _mol_weight(b) <= thr]
        else:
            blocks = blocks_all
    else:
        blocks = blocks_all
    # Intermediates become state_smiles; optionally filter by state MW
    interms_all = [i for i in parse_plus_list(row.get(COL_INTERMEDIATES)) if _is_valid_smiles(i)]
    if max_state_mw is not None:
        try:
            thr_s = float(max_state_mw)
        except Exception:
            thr_s = None
        if thr_s is not None and thr_s > 0:
            interms = [i for i in interms_all if _mol_weight(i) <= thr_s]
        else:
            interms = interms_all
    else:
        interms = interms_all
    template = _safe_str(row.get(COL_TEMPLATE, ""))

    route_id = row.get(COL_ROUTE_ID)
    step_index = row.get(COL_STEP_INDEX)
    route_score = row.get(COL_ROUTE_SCORE)

    out: List[dict] = []

    if interms:
        block_list = blocks if blocks else [""]
        for interm in interms:
            for block in block_list:
                out.append(
                    {
                        "ligand_smiles": ligand,
                        "route_id": route_id,
                        "step_index": step_index,
                        "route_score": route_score,
                        "state_smiles": interm,
                        "action_building_block": block,
                        "action_reaction_template": template,
                        "result_smiles": product,
                        "is_start_state": False,
                        "num_intermediate_candidates": len(interms),
                        "num_building_blocks": len(blocks),
                    }
                )
        return out

    # No intermediate -> starting step from only stock materials (Plan B semantics)
    # We want:
    #   state_smiles_0 = 单个起始砌块 (来自 building block 库)
    #   action_building_block = 另一个砌块（若存在；否则为空，表示单反应物模板）
    #   result_smiles = 第一个中间体产物
    if skip_start_steps:
        return []

    # 必须有至少一个有效的 building block
    if not blocks:
        return []

    # 去重以稳定选择
    uniq_blocks = []
    seen_blk = set()
    for b in blocks:
        if b not in seen_blk:
            seen_blk.add(b)
            uniq_blocks.append(b)

    # 选择根砌块：优先分子量最小的（更像核心小砌块）
    try:
        root_block = min(uniq_blocks, key=_mol_weight)
    except Exception:
        root_block = uniq_blocks[0]

    # 选择动作砌块：在剩余砌块中选一个（例如分子量最大的）
    remaining = [b for b in uniq_blocks if b != root_block]
    action_block = ""
    if remaining:
        try:
            action_block = max(remaining, key=_mol_weight)
        except Exception:
            action_block = remaining[0]

    out.append(
        {
            "ligand_smiles": ligand,
            "route_id": route_id,
            "step_index": step_index,
            "route_score": route_score,
            "state_smiles": root_block,
            "action_building_block": action_block,
            "action_reaction_template": template,
            "result_smiles": product,
            "is_start_state": True,
            "num_intermediate_candidates": 0,
            "num_building_blocks": len(uniq_blocks),
        }
    )
    return out


def convert(
    df: pd.DataFrame,
    skip_start_steps: bool,
    max_block_mw: float | None = None,
    max_state_mw: float | None = None,
    max_ligand_mw: float | None = None,
    log_every: int = 0,
) -> pd.DataFrame:
    rows: List[dict] = []
    total_rows = int(len(df))
    if total_rows > 0:
        print({"status": "conversion_start", "rows": total_rows, "max_block_mw": max_block_mw, "max_state_mw": max_state_mw, "max_ligand_mw": max_ligand_mw})
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        rows.extend(convert_row(r, skip_start_steps, max_block_mw, max_state_mw=max_state_mw, max_ligand_mw=max_ligand_mw))
        if log_every and i % int(max(1, log_every)) == 0:
            print({"conversion_progress": i, "total": total_rows})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["ligand_smiles", "route_id", "step_index"], kind="stable").reset_index(drop=True)
        # Normalize string columns to avoid NA propagation
        for col in [
            "ligand_smiles",
            "state_smiles",
            "action_building_block",
            "action_reaction_template",
            "result_smiles",
        ]:
            if col in out.columns:
                out[col] = out[col].fillna("")
    return out


def _prefer_non_empty_block_indices(g: pd.DataFrame, idxs: List[int]) -> int:
    """From candidate row indices within group g, pick the one that has a non-empty action_building_block.
    Tie-breaker: smallest step_index, then original DataFrame index for stability.
    """
    if not idxs:
        raise ValueError("No candidate indices provided")
    def key(i: int):
        row = g.loc[i]
        block = row.get("action_building_block")
        step = row.get("step_index")
        return (is_na_like(block), 10**9 if pd.isna(step) else int(step), i)
    return sorted(idxs, key=key)[0]


def assign_forward_order(out: pd.DataFrame, log_every_groups: int = 0) -> pd.DataFrame:
    """Assign forward step indices by linking rows where next.state == current.result.

    For each (ligand_smiles, route_id), we attempt to build a single longest chain
    starting from a plausible start (is_start_state or state not produced by any result),
    greedily following matches and preferring non-empty building blocks.

    Adds two columns:
    - forward_step_index (Int64): 1..N along the chosen chain; NA for rows off-chain
    - is_in_forward_chain (bool): True for rows on the chosen chain
    """
    if out.empty:
        out = out.copy()
        out["forward_step_index"] = pd.Series([], dtype="Int64")
        out["is_in_forward_chain"] = pd.Series([], dtype="bool")
        return out

    out = out.copy()
    out["forward_step_index"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    out["is_in_forward_chain"] = False

    group_keys = ["ligand_smiles", "route_id"]
    # Pre-compute number of groups for progress
    try:
        total_groups = int(out[group_keys].drop_duplicates().shape[0])
    except Exception:
        total_groups = 0
    if total_groups > 0:
        print({"status": "forward_order_start", "groups": total_groups})

    for gi, (_, g) in enumerate(out.groupby(group_keys, sort=False), start=1):
        if g.empty:
            continue
        # Work with index references into `out`
        idxs: List[int] = list(g.index)

        # Helper maps
        result_to_idxs: Dict[str, List[int]] = {}
        state_to_idxs: Dict[str, List[int]] = {}
        for i in idxs:
            rs = str(out.at[i, "result_smiles"]) if not is_na_like(out.at[i, "result_smiles"]) else ""
            ss = str(out.at[i, "state_smiles"]) if not is_na_like(out.at[i, "state_smiles"]) else ""
            result_to_idxs.setdefault(rs, []).append(i)
            state_to_idxs.setdefault(ss, []).append(i)

        all_results = set(result_to_idxs.keys())

        # Start candidates: is_start_state OR empty state OR state not produced by any result
        start_candidates: List[int] = []
        for i in idxs:
            ss = str(out.at[i, "state_smiles"]) if not is_na_like(out.at[i, "state_smiles"]) else ""
            is_start = bool(out.at[i, "is_start_state"]) if "is_start_state" in out.columns else False
            if is_start or is_na_like(ss) or (ss not in all_results):
                start_candidates.append(i)

        def start_key(i: int) -> Tuple[int, int, int]:
            is_start = bool(out.at[i, "is_start_state"]) if "is_start_state" in out.columns else False
            step = out.at[i, "step_index"]
            return (0 if is_start else 1, 10**9 if pd.isna(step) else int(step), i)

        start_candidates.sort(key=start_key)

        def build_chain(start_idx: int) -> List[int]:
            chain: List[int] = [start_idx]
            seen_pairs = set()
            # tuple key to prevent loops
            def pair_key(j: int) -> Tuple[str, str, str, str]:
                return (
                    str(out.at[j, "state_smiles"]) if not is_na_like(out.at[j, "state_smiles"]) else "",
                    str(out.at[j, "result_smiles"]) if not is_na_like(out.at[j, "result_smiles"]) else "",
                    str(out.at[j, "action_building_block"]) if not is_na_like(out.at[j, "action_building_block"]) else "",
                    str(out.at[j, "action_reaction_template"]) if not is_na_like(out.at[j, "action_reaction_template"]) else "",
                )
            seen_pairs.add(pair_key(start_idx))

            current_idx = start_idx
            while True:
                curr_res = str(out.at[current_idx, "result_smiles"]) if not is_na_like(out.at[current_idx, "result_smiles"]) else ""
                next_idxs = state_to_idxs.get(curr_res, [])
                if not next_idxs:
                    break
                # Prefer non-empty block; tie-break by step_index
                try:
                    chosen_idx = _prefer_non_empty_block_indices(out.loc[idxs], next_idxs)
                except Exception:
                    # Fallback: pick smallest step_index
                    chosen_idx = sorted(next_idxs, key=lambda j: (10**9 if pd.isna(out.at[j, "step_index"]) else int(out.at[j, "step_index"]), j))[0]
                pk = pair_key(chosen_idx)
                if pk in seen_pairs:
                    break
                seen_pairs.add(pk)
                chain.append(chosen_idx)
                current_idx = chosen_idx
            return chain

        best_chain: List[int] = []
        for sidx in start_candidates or idxs:
            chain = build_chain(sidx)
            if len(chain) > len(best_chain):
                best_chain = chain

        # Assign indices
        for rank, row_idx in enumerate(best_chain, start=1):
            out.at[row_idx, "forward_step_index"] = int(rank)
            out.at[row_idx, "is_in_forward_chain"] = True

        if log_every_groups and gi % int(max(1, log_every_groups)) == 0:
            print({"forward_order_progress": gi, "total_groups": total_groups})

    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert retrosynthesis dataset to forward one-step trajectories")
    p.add_argument("--input", type=str, default="data/reaction_paths_all_routes.csv", help="Input CSV path")
    p.add_argument("--output", type=str, default="data/forward_trajectories.csv", help="Output CSV path")
    p.add_argument("--skip-start-steps", action="store_true", help="Skip steps with no intermediate (start state)")
    p.add_argument("--max-block-mw", type=float, default=200.0, help="Maximum molecular weight for building block reactants (Da)")
    p.add_argument("--max-state-mw", type=float, default=None, help="Maximum molecular weight for state_smiles (Da)")
    p.add_argument("--max-ligand-mw", type=float, default=None, help="Maximum molecular weight for ligand_smiles (Da)")
    p.add_argument("--log-every", type=int, default=10000, help="Log conversion progress every N input rows (0=off)")
    p.add_argument("--log-every-groups", type=int, default=100, help="Log forward-order progress every N (ligand,route) groups (0=off)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    df = load_dataset(args.input)
    print({"input_loaded": len(df), "input_path": args.input})
    out = convert(
        df,
        skip_start_steps=args.skip_start_steps,
        max_block_mw=args.max_block_mw,
        max_state_mw=getattr(args, "max_state_mw", None),
        max_ligand_mw=getattr(args, "max_ligand_mw", None),
        log_every=int(getattr(args, "log_every", 0)),
    )
    out = assign_forward_order(out, log_every_groups=int(getattr(args, "log_every_groups", 0)))
    # Save without introducing textual 'nan' markers
    out.to_csv(args.output, index=False)
    print(
        {
            "saved": len(out),
            "routes": int(out["route_id"].nunique()) if not out.empty else 0,
            "ligands": int(out["ligand_smiles"].nunique()) if not out.empty else 0,
            "start_steps": int(out["is_start_state"].sum()) if not out.empty else 0,
            "chain_steps": int(out["is_in_forward_chain"].sum()) if not out.empty else 0,
            "output": args.output,
        }
    )


if __name__ == "__main__":
    main()




