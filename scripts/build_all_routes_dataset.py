# 用于构建所有ligand合成路径数据集
import os
import csv
import argparse
from typing import List, Dict, Any, Optional

import pandas as pd
from aizynthfinder.aizynthfinder import AiZynthFinder
from SynthPolicyNet.data_utils import mol_from_smiles


def extract_route_steps(route_tree: Dict[str, Any], max_stock_mw: Optional[float] = None) -> List[Dict[str, str]]:
    """
    Extract steps from a single route tree dict in forward order.

    For each reaction node under a molecule node, emit one record:
      - 当前状态分子: the product molecule (parent molecule) SMILES
      - 在zinc库里的反应物砌块分子: joined SMILES of in-stock reactants
      - 和反应物砌块分子反应的中间体分子: joined SMILES of intermediate reactants
      - 反应模版: reaction template/reaction_name (metadata from reaction node)
    The returned list is ordered from starting materials to the target molecule.
    """
    steps: List[Dict[str, str]] = []

    def _is_valid_smiles(smiles: str) -> bool:
        try:
            return bool(mol_from_smiles(smiles))
        except Exception:
            return False

    def _mol_weight(smiles: str) -> float:
        try:
            mol = mol_from_smiles(smiles)
            if not mol:
                return float("inf")
            from rdkit.Chem import Descriptors  # type: ignore
            return float(Descriptors.MolWt(mol))
        except Exception:
            return float("inf")

    def _walk_molecule_node(mol_node: Dict[str, Any]):
        if not isinstance(mol_node, dict):
            return
        # Only consider molecule nodes (presence of in_stock)
        if "in_stock" not in mol_node:
            return
        # Leaf node (in stock) has no reaction
        if mol_node.get("in_stock", False):
            return

        product_smiles = mol_node.get("smiles", "")
        if not product_smiles:
            return
        # Skip steps where product SMILES is invalid
        if not _is_valid_smiles(product_smiles):
            return

        # Children of a molecule node are reaction nodes
        reaction_nodes = [child for child in (mol_node.get("children", []) or []) if isinstance(child, dict) and "in_stock" not in child]
        for rxn_node in reaction_nodes:
            # Children of a reaction node are reactant molecule nodes
            reactant_mol_nodes = [c for c in (rxn_node.get("children", []) or []) if isinstance(c, dict) and "in_stock" in c]

            # Recurse into intermediate reactants first (bottom-up)
            for interm_node in [c for c in reactant_mol_nodes if not c.get("in_stock", False)]:
                _walk_molecule_node(interm_node)

            in_stock_reactants = [c.get("smiles", "") for c in reactant_mol_nodes if c.get("in_stock", False)]
            intermediate_reactants = [c.get("smiles", "") for c in reactant_mol_nodes if not c.get("in_stock", False)]

            # Filter out invalid SMILES in reactants
            in_stock_reactants = [s for s in in_stock_reactants if s and _is_valid_smiles(s)]
            intermediate_reactants = [s for s in intermediate_reactants if s and _is_valid_smiles(s)]

            # Optional: filter in-stock building blocks by molecular weight threshold
            if max_stock_mw is not None:
                try:
                    thr = float(max_stock_mw)
                except Exception:
                    thr = None
                if thr is not None and thr > 0:
                    in_stock_reactants = [s for s in in_stock_reactants if _mol_weight(s) <= thr]

            rxn_meta = rxn_node.get("metadata", {}) or {}
            template = rxn_meta.get("template") or rxn_meta.get("reaction_name") or "N/A"

            steps.append({
                "当前状态分子": product_smiles,
                "在zinc库里的反应物砌块分子": " + ".join([s for s in in_stock_reactants if s]) if in_stock_reactants else "N/A",
                "和反应物砌块分子反应的中间体分子": " + ".join([s for s in intermediate_reactants if s]) if intermediate_reactants else "N/A",
                "反应模版": template,
            })

    root = route_tree if isinstance(route_tree, dict) else {}
    if isinstance(root, dict) and root.get("smiles"):
        _walk_molecule_node(root)

    # steps are collected bottom-up; reverse to forward order
    steps.reverse()
    return steps


def main():
    parser = argparse.ArgumentParser(description="Extract all retrosynthesis routes for a SMILES list to CSV, grouped by ligand and route.")
    parser.add_argument("--config", default="config.yml", help="Path to AiZynthFinder config.yml")
    parser.add_argument("--input", default="data/protein_ligand_pactivity.csv", help="Input CSV with column 'ligand_smiles'")
    parser.add_argument("--limit", type=int, default=20, help="Number of unique SMILES to process from the top")
    parser.add_argument("--stock", default="zinc", help="Stock name as defined in config.yml")
    parser.add_argument("--expansion-policy", dest="expansion_policy", default="uspto", help="Expansion policy name")
    parser.add_argument("--filter-policy", dest="filter_policy", default="uspto", help="Filter policy name")
    parser.add_argument("--output", default="data/reaction_paths_all_routes.csv", help="Output CSV path")
    parser.add_argument("--max-stock-mw", dest="max_stock_mw", type=float, default=200.0, help="Maximum molecular weight for in-stock building blocks (Da)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件未找到: {args.config}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件未找到: {args.input}")

    # 读取并去重 SMILES
    df_in = pd.read_csv(args.input)
    if "ligand_smiles" not in df_in.columns:
        raise ValueError("输入 CSV 缺少列 'ligand_smiles'")
    smiles_list = (
        df_in["ligand_smiles"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().drop_duplicates().tolist()
    )
    if args.limit and args.limit > 0:
        smiles_list = smiles_list[: args.limit]

    print(f"将处理 {len(smiles_list)} 个唯一 SMILES（最多 {args.limit} 个）")

    print("初始化 AiZynthFinder...")
    finder = AiZynthFinder(configfile=args.config)
    finder.stock.select(args.stock)
    finder.expansion_policy.select(args.expansion_policy)
    finder.filter_policy.select(args.filter_policy)

    # Streamed writing: accumulate rows then flush in chunks to avoid high memory usage
    CHUNK_SIZE = 5000
    buffer_rows: List[Dict[str, str]] = []
    total_written = 0
    seen_route_ids = set()

    # Prepare output CSV with header
    headers = [
        "ligand_smiles",
        "解析树ID",
        "步骤序号",
        "路线分数",
        "当前状态分子",
        "在zinc库里的反应物砌块分子",
        "和反应物砌块分子反应的中间体分子",
        "反应模版",
    ]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

    def flush_buffer():
        nonlocal buffer_rows, total_written
        if not buffer_rows:
            return
        with open(args.output, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerows(buffer_rows)
        total_written += len(buffer_rows)
        buffer_rows = []

    for lig_index, target_smiles in enumerate(smiles_list, start=1):
        print(f"[{lig_index}/{len(smiles_list)}] 目标分子: {target_smiles}")
        try:
            finder.target_smiles = target_smiles
            finder.tree_search()
            finder.build_routes()
        except Exception as e:
            print(f"  - 搜索失败，跳过。错误: {e}")
            continue

        route_dicts = getattr(finder.routes, "dicts", [])
        print(f"  - 找到完整反应路线数: {len(route_dicts)}")
        if not route_dicts:
            continue

        for ridx, route in enumerate(route_dicts, start=1):
            route_id = f"lig{lig_index}_route_{ridx}"
            score_total = None
            try:
                score_total = float(route.get("score", {}).get("total"))
            except Exception:
                score_total = None

            # 与 dataset_builder.py 对齐：优先取 route['tree']，若无则回退到整个字典
            tree = route.get("tree") if isinstance(route, dict) else None
            if tree is None:
                tree = route if isinstance(route, dict) else {}

            steps = extract_route_steps(tree, max_stock_mw=args.max_stock_mw)
            for step_order, step in enumerate(steps, start=1):
                row = {
                    "ligand_smiles": target_smiles,
                    "解析树ID": route_id,
                    "步骤序号": step_order,
                    "路线分数": f"{score_total:.4f}" if isinstance(score_total, float) else "N/A",
                    **step,
                }
                buffer_rows.append(row)
                if len(buffer_rows) >= CHUNK_SIZE:
                    flush_buffer()

    # Final flush any remaining rows
    flush_buffer()

    if total_written == 0:
        print("未能从任何路线解析出步骤。")
        return

    # 汇总信息（使用计数器与集合，避免将全部行驻留内存）
    try:
        # If we tracked route IDs while iterating, use that; else compute from file cheaply
        if not seen_route_ids:
            # Fallback: compute unique route ids by streaming
            uniq = set()
            for chunk in pd.read_csv(args.output, usecols=["解析树ID"], chunksize=100000):
                uniq.update(chunk["解析树ID"].dropna().astype(str).unique().tolist())
            seen_route_ids = uniq
        total_routes = len(seen_route_ids)
    except Exception:
        total_routes = -1
    print(f"已写入: {args.output}，共 {total_written} 条步骤，{total_routes} 条解析树，自 {len(smiles_list)} 个 ligand_smiles。")


if __name__ == "__main__":
    main()


