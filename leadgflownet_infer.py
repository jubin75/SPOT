from __future__ import annotations

import argparse
import json
import os
import sys
import random
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import torch
from torch import nn

from SynthPolicyNet.train_policy import build_forward_dataset
from SynthPolicyNet.data_utils import build_graph_from_smiles, get_atom_feature_dim
from SynthPolicyNet.datasets import Vocab
from SynthPolicyNet.models import SynthPolicyNet
from LeadGFlowNet.conditional_policy import ConditionalSynthPolicy
from LeadGFlowNet.protein_encoder import SimpleProteinEncoder, Esm2ProteinEncoder, tokenize_protein
from LeadGFlowNet.qsar import QSARPredictor
from LeadGFlowNet.template_expander import TemplateLibrary
from rdkit import Chem
from rdkit.Chem import AllChem, QED, FilterCatalog
from rdkit import DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
try:
    RDLogger.DisableLog('rdApp.error')
except Exception:
    pass
import numpy as np


@dataclass
class Edge:
    block_smiles: str
    rxn_template: str
    result_smiles: str


def build_route_graph(df) -> Tuple[Dict[str, List[Edge]], List[str]]:
    """Build adjacency lists of feasible transitions from forward trajectories CSV.

    Returns:
        graph: mapping state_smiles -> list of Edge
        start_states: list of plausible start states.

    Plan B 语义下（forward_trajectories_planB.csv）：
    - 优先把 `is_start_state == True` 的 state_smiles 当作起始根节点，
      这些正是“单个起始 building block”（来自库）；
    - 若无 is_start_state 列或该列全为 False，则回退到旧逻辑：
      选取那些从未作为 result_smiles 出现的 state_smiles 作为起点。
    """
    graph: Dict[str, List[Edge]] = {}
    all_states: set[str] = set()
    produced: set[str] = set()

    for _, row in df.iterrows():
        state = str(row.get("state_smiles", ""))
        block = str(row.get("action_building_block", ""))
        rxn = str(row.get("action_reaction_template", ""))
        result = str(row.get("result_smiles", ""))
        all_states.add(state)
        produced.add(result)
        if state not in graph:
            graph[state] = []
        graph[state].append(Edge(block_smiles=block, rxn_template=rxn, result_smiles=result))

    # Plan B: 优先使用 is_start_state 标注的 state_smiles 作为起始节点（单一砌块根）
    start_states: List[str] = []
    if "is_start_state" in df.columns:
        try:
            mask = df["is_start_state"].astype(bool)
            if mask.any():
                vals = df.loc[mask, "state_smiles"].astype(str).tolist()
                # 去重并去掉空字符串
                start_states = sorted({s for s in vals if s})
        except Exception:
            start_states = []

    # 回退：兼容旧数据集（无 is_start_state）或标注缺失的情况
    if not start_states:
        start_states = [s for s in all_states if s and (s not in produced)]
        # Fallback: if empty states exist and no explicit starts, add empty string as a start
        if not start_states and ("" in graph):
            start_states = [""]
    return graph, start_states


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[Dict[str, Any], Optional[Vocab], Optional[Vocab]]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    obj = torch.load(checkpoint_path, map_location=device)
    block_vocab = Vocab.from_json(obj["block_vocab"]) if "block_vocab" in obj else None
    rxn_vocab = Vocab.from_json(obj["rxn_vocab"]) if "rxn_vocab" in obj else None
    return obj, block_vocab, rxn_vocab


def build_block_graphs(block_vocab: Vocab) -> List:
    node_dim = get_atom_feature_dim()
    graphs: List = []
    for s in block_vocab.itos:
        g = build_graph_from_smiles(s)
        if g is None:
            # Minimal 1-node graph placeholder
            import torch_geometric
            from torch_geometric.data import Data  # type: ignore

            graphs.append(Data(x=torch.zeros((1, node_dim), dtype=torch.float32), edge_index=torch.zeros((2, 0), dtype=torch.long)))
        else:
            graphs.append(g)
    return graphs


def softmax_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        temperature = 1.0
    return torch.softmax(logits / float(temperature), dim=-1)


def _nucleus_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    p = max(0.0, min(1.0, float(p)))
    if p <= 0.0:
        return probs
    vals, idxs = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(vals, dim=-1)
    k = int(torch.searchsorted(cumsum, torch.tensor(p, device=probs.device)).item()) + 1
    k = max(1, min(k, probs.numel()))
    mask = torch.zeros_like(probs)
    mask[idxs[:k]] = 1.0
    return probs * mask


def _fp_ecfp4(smiles: str, n_bits: int = 2048):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=n_bits)
    except Exception:
        return None


def _tanimoto(a, b) -> float:
    try:
        return float(DataStructs.TanimotoSimilarity(a, b))
    except Exception:
        return 0.0


def sample_topk_indices(probs: torch.Tensor, topk: int) -> List[int]:
    topk = max(1, int(topk))
    K = min(topk, probs.shape[-1])
    vals, idxs = torch.topk(probs, k=K, dim=-1)
    return idxs.detach().cpu().tolist()


def _sanitize_probs(probs: torch.Tensor) -> torch.Tensor:
    """Clamp to valid probability vector on the same device/dtype.

    - Replace NaN/Inf with 0
    - Clamp negatives to 0
    - Renormalize to sum==1; if all-zero, return uniform
    """
    p = probs
    # Ensure tensor type is floating
    if not torch.is_floating_point(p):
        p = p.float()
    # Replace non-finite with 0
    finite = torch.isfinite(p)
    if not bool(finite.all()):
        p = torch.where(finite, p, torch.zeros_like(p))
    # Clamp negatives
    p = torch.clamp(p, min=0.0)
    s = torch.sum(p)
    if not torch.isfinite(s) or float(s.item()) <= 0.0:
        # uniform
        N = max(1, int(p.numel()))
        p = torch.ones_like(p) / float(N)
    else:
        p = p / float(s.item())
    return p


def main() -> None:
    p = argparse.ArgumentParser(description="Sample synthesis route trees using a trained ConditionalSynthPolicy")
    p.add_argument("--input", default="data/reaction_paths_all_routes.csv", help="Retrosynthesis CSV path")
    p.add_argument("--forward", default="data/forward_trajectories.csv", help="Forward trajectories CSV path")
    p.add_argument("--rebuild-forward", action="store_true", help="Rebuild forward CSV even if exists")
    p.add_argument("--checkpoint", default="checkpoints/synth_policy_net.pt")
    p.add_argument("--use-cpu", action="store_true")
    p.add_argument("--protein", type=str, required=True, help="Target protein sequence")
    p.add_argument("--protein-encoder", type=str, default="esm2", choices=["simple", "esm2"], help="Protein encoder backend")
    p.add_argument("--esm2-model", type=str, default="lib/models--facebook--esm2_t30_150M_UR50D", help="Local ESM2 model path or HF id")
    p.add_argument("--num-samples", type=int, default=1000, help="Number of independent trees to sample")
    p.add_argument("--max-depth", type=int, default=10, help="Maximum expansion depth of each tree")
    p.add_argument("--branch-block-topk", type=int, default=2, help="At each state, branch on top-K blocks")
    p.add_argument("--branch-rxn-topk", type=int, default=1, help="For each chosen block, branch on top-K rxns")
    p.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for sampling")
    p.add_argument("--output-json", default="runs/lead_routes.json")
    p.add_argument("--progress-file", type=str, default="", help="Append the current sampled route index (1-based) after each sample")
    # QSAR-guided search
    p.add_argument("--use-qsar", action="store_true", help="Enable QSAR-guided search ordering")
    p.add_argument("--qsar-checkpoint", type=str, default="checkpoints/qsar.pt", help="QSAR checkpoint path")
    p.add_argument("--qsar-mix", type=float, default=1.0, help="If >0, sort children by (policy_prob^(1-qsar_mix)) * (sigmoid(qsar)^(qsar_mix))")
    # QSAR-ranked export
    p.add_argument("--export-ranked", action="store_true", help="Export QSAR-ranked leads (CSV/JSON) alongside output JSON")
    p.add_argument("--output-ranked-csv", type=str, default="runs/leads_qsar_ranked.csv", help="CSV path for QSAR-ranked leads")
    p.add_argument("--output-ranked-json", type=str, default="runs/leads_qsar_ranked.json", help="JSON path for QSAR-ranked leads")
    p.add_argument("--min-qsar", type=float, default=0.0, help="If >0, filter ranked outputs to QSAR(sigmoid) >= this threshold")
    # Optional: filter CSV by a Vina affinity threshold (keep entries with affinity < threshold)
    p.add_argument("--filter-vina-th", type=float, default=None, help="Only write rows whose vina_affinity < this threshold to CSV (e.g., -7.0)")
    # Optional: include PLANTAIN scores in ranked CSV
    p.add_argument("--use-plantain", action="store_true", help="Compute PLANTAIN min score for ranked CSV/JSON entries")
    p.add_argument("--plantain-pocket", type=str, default="", help="Pocket PDB path for PLANTAIN; if empty, auto-detect under test/<PDBID>/*_pocket.pdb")
    p.add_argument("--plantain-device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"], help="Device for PLANTAIN scoring")
    p.add_argument("--plantain-poses-dir", type=str, default="runs/plantain_poses", help="Directory to save PLANTAIN pose SDFs for debug/inspection")
    # Vina refine (Plantain+Vina) for ranked export
    p.add_argument("--use-vina", action="store_true", help="Rescore PLANTAIN poses with python-vina and export vina_affinity")
    p.add_argument("--vina-pdbqt-dir", type=str, default="runs/vina_pdbqt", help="Where to write ligand PDBQT files")
    p.add_argument("--vina-obabel-bin", type=str, default="/usr/local/bin/obabel", help="Path to obabel binary")
    p.add_argument("--vina-center", type=str, default="", help="Override Vina grid center as 'cx,cy,cz'")
    p.add_argument("--vina-box-size", type=float, default=22.0, help="Cubic Vina box size (A)")
    p.add_argument("--vina-exhaustiveness", type=int, default=32, help="Dock exhaustiveness for local refine")
    p.add_argument("--vina-top-k", type=int, default=3, help="Top-K PLANTAIN poses to refine per ligand")
    p.add_argument("--vina-full-dock-th", type=float, default=-3.0, help="If minimized affinity > th, run a quick dock")
    p.add_argument("--vina-strict", action="store_true", help="Require ADFR/Meeko; error if unavailable (no obabel fallback)")
    # Sampling diversity controls
    p.add_argument("--sampling-method", type=str, default="default", choices=["default", "nucleus"], help="Sampling method for block/rxn selection")
    p.add_argument("--nucleus-p", type=float, default=0.9, help="Cumulative mass threshold for nucleus sampling")
    # Post-selection diversity
    p.add_argument("--select-k", type=int, default=200, help="Select at most K diverse leads from the generated set")
    p.add_argument("--diversity-mode", type=str, default="mmr", choices=["none", "minsim", "mmr"], help="Diverse selection mode")
    p.add_argument("--minsim-th", type=float, default=0.5, help="Greedy min-sim threshold when diversity-mode=minsim")
    p.add_argument("--mmr-lambda", type=float, default=0.7, help="MMR tradeoff between score and novelty (0..1)")
    # Stochasticity & seeds
    p.add_argument("--deterministic", action="store_true", help="Disable stochastic sampling; use deterministic top-k")
    p.add_argument("--seed", type=int, default=-1, help="Random seed; <0 uses system randomness")
    # Open-space (template/free-walk) inference
    p.add_argument("--template-walk", action="store_true", help="Enable template-guided open-space expansion at inference")
    p.add_argument("--template-csv", type=str, default="", help="Template library CSV/XLSX path")
    p.add_argument("--template-max-rows", type=int, default=1500, help="Max rows to load from template CSV")
    p.add_argument("--template-try-templates", type=int, default=32, help="Per-state max templates to try for expansion")
    p.add_argument("--template-sample-blocks", type=int, default=64, help="Per-state max external blocks to sample for two-reactant templates")
    p.add_argument("--extra-blocks-csv", type=str, default="data/building_blocks_frag_mw250.csv", help="CSV with external block SMILES (column name flexible)")
    p.add_argument("--extra-blocks-cap", type=int, default=5000, help="Cap of external block rows to load")
    p.add_argument("--feasibility-filter", type=str, default="rdkit", choices=["none", "rdkit"], help="Feasibility filter for open-space products")
    p.add_argument("--open-max-proposals", type=int, default=8, help="Max open-space children to add per state")
    # Step constraints and shaping
    p.add_argument(
        "--template-physical-filter",
        dest="template_physical_filter",
        action="store_true",
        help="For template-walk proposals, enforce heavy-atom growth (and cap growth) vs current state",
    )
    p.add_argument(
        "--template-min-delta-heavy",
        type=int,
        default=1,
        help="For template-walk proposals: minimum allowed heavy-atom increase vs current state (<=0 disables growth requirement)",
    )
    p.add_argument(
        "--template-max-delta-heavy",
        type=int,
        default=40,
        help="For template-walk proposals: maximum allowed heavy-atom increase vs current state (<=0 disables upper cap)",
    )
    p.add_argument(
        "--single-react-max-delta-heavy",
        type=int,
        default=40,
        help="Maximum allowed heavy-atom increase for single-reactant steps (block_smiles empty); <=0 disables this cap",
    )
    p.add_argument("--no-immediate-undo", dest="no_immediate_undo", action="store_true", help="Disallow children that reduce heavy atoms vs current state")
    p.add_argument("--allow-immediate-undo", dest="no_immediate_undo", action="store_false")
    p.add_argument(
        "--step-plantain",
        dest="step_plantain",
        action="store_true",
        help="Re-rank children by PLANTAIN(min) every interval for top-M",
    )
    p.add_argument(
        "--no-step-plantain",
        dest="step_plantain",
        action="store_false",
        help="Disable per-step PLANTAIN re-ranking (keep PLANTAIN only for final filtering/export)",
    )
    p.add_argument("--step-plantain-interval", type=int, default=1, help="Apply step PLANTAIN re-rank every N expansions")
    p.add_argument("--step-plantain-topk", type=int, default=8, help="Evaluate at most top-M children with PLANTAIN per application")
    p.add_argument("--step-plantain-mix", type=float, default=0.3, help="Mixing weight for policy vs PLANTAIN: combined = policy^(1-mix) * exp(-score/scale)^mix")
    p.set_defaults(no_immediate_undo=True, step_plantain=True, template_physical_filter=True)
    # Expansion strategy
    p.add_argument("--expand-mode", type=str, default="path", choices=["tree", "path"], help="'tree' builds full K^D tree; 'path' samples one branch per depth (much faster)")
    args = p.parse_args()

    # Device
    if args.use_cpu:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Dataframe and route graph
    fwd_df = build_forward_dataset(args.input, args.forward, skip_start_steps=True, rebuild=args.rebuild_forward)
    route_graph, start_states = build_route_graph(fwd_df)
    if not start_states:
        raise RuntimeError("No start states found in forward trajectories. Consider rebuilding with start steps included.")

    # Load checkpoint + model + vocabs
    obj, ckpt_block_vocab, ckpt_rxn_vocab = load_checkpoint(args.checkpoint, device)

    # Build model consistent with checkpoint vocabs
    # Robustly infer dims from checkpoint if metadata is missing
    hidden_dim = int(obj.get("hidden_dim", -1))
    num_gnn_layers = int(obj.get("num_gnn_layers", -1))
    sd0 = obj.get("model_state", obj)
    if hidden_dim <= 0:
        try:
            w = sd0.get("state_encoder.convs.0.lin.weight") if isinstance(sd0, dict) else None
            if w is not None and hasattr(w, "shape"):
                hidden_dim = int(w.shape[0])
        except Exception:
            hidden_dim = -1
    if num_gnn_layers <= 0:
        try:
            if isinstance(sd0, dict):
                layer_count = sum(1 for k in sd0.keys() if k.startswith("state_encoder.convs.") and k.endswith(".lin.weight"))
                if layer_count > 0:
                    num_gnn_layers = int(layer_count)
        except Exception:
            num_gnn_layers = -1
    if hidden_dim <= 0:
        hidden_dim = 256
    if num_gnn_layers <= 0:
        num_gnn_layers = 3
    if ckpt_block_vocab is None or ckpt_rxn_vocab is None:
        # Fallback to vocabs from data
        # Note: this may not exactly match training; strongly prefer checkpoints with saved vocabs
        from SynthPolicyNet.datasets import ForwardTrajectoryDataset

        dataset_tmp = ForwardTrajectoryDataset(
            fwd_df,
            block_vocab=None,
            rxn_vocab=None,
            use_only_forward_chain=True,
            skip_start_states=True,
        )
        ckpt_block_vocab = dataset_tmp.block_vocab
        ckpt_rxn_vocab = dataset_tmp.rxn_vocab

    model = SynthPolicyNet(
        node_feature_dim=get_atom_feature_dim(),
        hidden_dim=hidden_dim,
        num_building_blocks=len(ckpt_block_vocab.itos),
        num_reaction_templates=len(ckpt_rxn_vocab.itos),
        num_gnn_layers=num_gnn_layers,
        dropout=0.0,
        share_encoders=False,
    ).to(device)
    model.load_state_dict(obj["model_state"], strict=False)
    model.eval()

    # Conditional policy
    if args.protein_encoder == "esm2":
        prot_enc = Esm2ProteinEncoder(model_name=args.esm2_model).to(device)
        protein_emb = prot_enc.encode_sequence(args.protein)
        protein_dim = prot_enc.out_dim
        if protein_dim != (hidden_dim // 2):
            protein_adapter = nn.Linear(protein_dim, hidden_dim // 2).to(device)
            protein_emb = protein_adapter(protein_emb)
    else:
        prot_enc = SimpleProteinEncoder(embed_dim=hidden_dim // 2, lstm_hidden=hidden_dim // 2).to(device)
        token_ids = tokenize_protein(args.protein).to(device)
        protein_emb = prot_enc(token_ids)

    cond_policy = ConditionalSynthPolicy(model, protein_dim=protein_emb.shape[-1]).to(device)

    # Block embeddings
    block_graphs = build_block_graphs(ckpt_block_vocab)
    block_embs = model.encode_blocks(block_graphs, device=device)

    # Utility: map smiles/template to indices
    block_to_idx = ckpt_block_vocab.stoi
    rxn_to_idx = ckpt_rxn_vocab.stoi
    idx_to_block = ckpt_block_vocab.itos
    idx_to_rxn = ckpt_rxn_vocab.itos

    # QSAR predictor (optional)
    qsar_predictor: Optional[QSARPredictor] = None
    if bool(args.use_qsar):
        if not os.path.exists(args.qsar_checkpoint):
            raise FileNotFoundError(f"QSAR checkpoint not found: {args.qsar_checkpoint}")
        qsar_predictor = QSARPredictor(args.qsar_checkpoint, device=device)

    # Auto-detect Plantain pocket if requested but not provided
    plantain_pocket = str(getattr(args, "plantain_pocket", ""))
    use_plantain = bool(getattr(args, "use_plantain", False))
    if use_plantain and not plantain_pocket:
        try:
            import glob as _glob
            import os as _os
            tests = sorted([d for d in _glob.glob("test/*") if _os.path.isdir(d)])
            for d in tests:
                cand = sorted(_glob.glob(_os.path.join(d, "*_pocket.pdb")))
                if cand:
                    plantain_pocket = cand[0]
                    break
        except Exception:
            plantain_pocket = ""

    # Caches for open-space (template) proposals
    template_lib: Optional[TemplateLibrary] = None
    extra_blocks: List[str] = []
    open_cache: Dict[str, List[Any]] = {}
    block_library_meta: Dict[str, Dict[str, Optional[str]]] = {}
    extra_blocks_loaded = False

    def _normalize_meta_value(val: Optional[Any]) -> Optional[str]:
        if val is None:
            return None
        if isinstance(val, float):
            if math.isnan(val):
                return None
        s = str(val).strip()
        if not s:
            return None
        if s.lower() in {"", "nan", "none", "null", "<na>", "na"}:
            return None
        return s

    def _ensure_block_library_loaded() -> None:
        nonlocal extra_blocks_loaded, extra_blocks, block_library_meta
        if extra_blocks_loaded:
            return
        extra_blocks_loaded = True
        path = str(getattr(args, "extra_blocks_csv", "") or "").strip()
        if not path or not os.path.exists(path):
            return
        try:
            import pandas as pd
        except Exception:
            return
        cap = int(getattr(args, "extra_blocks_cap", 0))
        read_kwargs = {"dtype": str, "low_memory": False}
        try:
            dfb = pd.read_csv(path, **read_kwargs)
        except Exception:
            return
        if dfb.empty:
            return

        def _find_col(keywords: List[str]) -> Optional[str]:
            for kw in keywords:
                for col in dfb.columns:
                    if kw in str(col).lower():
                        return col
            return None

        col_smiles = _find_col(["smiles", "smile", "block"])
        if col_smiles is None and len(dfb.columns) > 0:
            col_smiles = dfb.columns[0]
        col_id = _find_col(["id", "catalog", "identifier"])
        col_size = _find_col(["size", "amount", "weight"])
        col_price = _find_col(["price", "cost"])
        if col_smiles is None:
            return

        seen: List[str] = []
        for _, row in dfb.iterrows():
            smi = _normalize_meta_value(row.get(col_smiles))
            if not smi:
                continue
            seen.append(smi)
            block_library_meta[smi] = {
                "smiles": smi,
                "id": _normalize_meta_value(row.get(col_id)) if col_id else None,
                "size": _normalize_meta_value(row.get(col_size)) if col_size else None,
                "price": _normalize_meta_value(row.get(col_price)) if col_price else None,
            }
        if seen:
            dedup: Dict[str, None] = {}
            for smi in seen:
                if smi not in dedup:
                    dedup[smi] = None
            all_blocks = list(dedup.keys())
            if cap > 0 and len(all_blocks) > cap:
                extra_blocks = all_blocks[:cap]
            else:
                extra_blocks = all_blocks

    _ensure_block_library_loaded()

    # Sampling helpers
    def compute_state_block_embeddings(state_smiles_list: List[str]) -> torch.Tensor:
        # Build minimal graphs for states and encode with the model's state encoder via cond_policy
        from torch_geometric.data import Data

        graphs = []
        node_dim = get_atom_feature_dim()
        for s in state_smiles_list:
            g = build_graph_from_smiles(s)
            if g is None:
                g = Data(x=torch.zeros((1, node_dim), dtype=torch.float32), edge_index=torch.zeros((2, 0), dtype=torch.long))
            # Ensure a batch vector exists (single graph -> all zeros)
            if not hasattr(g, "batch") or g.batch is None:
                g.batch = torch.zeros((g.x.size(0),), dtype=torch.long)
            graphs.append(g)

        # Batch graphs using simple concat; cond_policy expects a PyG batch, but its encoder supports Data with .batch
        # For simplicity, process one by one here
        hs: List[torch.Tensor] = []
        for g in graphs:
            g = g.to(device)
            # Single-graph batch dimension emulation
            # Compute conditioned state embedding for this single graph
            h = cond_policy.compute_h_state_block(g, protein_emb)  # (1, D)
            hs.append(h)
        return torch.cat(hs, dim=0)

    step_counter = {"n": 0}

    def _heavy_atoms(s: str) -> int:
        try:
            m = Chem.MolFromSmiles(s)
            return int(m.GetNumHeavyAtoms()) if m is not None else -1
        except Exception:
            return -1

    def _is_valid_smiles(s: str) -> bool:
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                return False
            # Sanitize to catch valence and charge issues early
            Chem.SanitizeMol(m)
            return True
        except Exception:
            return False

    def expand_state(state: str, max_children: Tuple[int, int]) -> List[Dict[str, Any]]:
        edges = route_graph.get(state, [])
        if not edges and not bool(getattr(args, "template_walk", False)):
            return []
        # Compute action distributions
        h_state = compute_state_block_embeddings([state])  # (1, D)
        block_logits = model.compute_block_logits(h_state, block_embs)  # (1, N)
        block_probs = softmax_logits(block_logits.squeeze(0), temperature=args.temperature)
        if args.sampling_method == "nucleus":
            block_probs = _nucleus_filter(block_probs, args.nucleus_p)
        block_probs = _sanitize_probs(block_probs)
        # Feasible-set mask for blocks at this state (when dataset edges exist)
        allowed_block_idxs: set[int] = set()
        if edges:
            for e in edges:
                bi = block_to_idx.get(e.block_smiles)
                if bi is not None:
                    allowed_block_idxs.add(int(bi))
            if allowed_block_idxs:
                maskb = torch.zeros_like(block_probs)
                for bi in allowed_block_idxs:
                    if 0 <= bi < maskb.numel():
                        maskb[bi] = 1.0
                block_probs = block_probs * maskb
                block_probs = _sanitize_probs(block_probs)
        sumb = float(torch.sum(block_probs).item())
        if sumb > 0:
            block_probs = block_probs / sumb
        else:
            # Fallback: uniform over feasible blocks if available, else uniform over all
            if allowed_block_idxs:
                block_probs = torch.zeros_like(block_probs)
                for bi in allowed_block_idxs:
                    if 0 <= bi < block_probs.numel():
                        block_probs[bi] = 1.0
                block_probs = _sanitize_probs(block_probs)
            else:
                block_probs = torch.ones_like(block_probs) / float(block_probs.numel())
        block_probs = _sanitize_probs(block_probs)
        # stochastic or deterministic selection of block indices
        if not bool(getattr(args, "deterministic", False)):
            Kb = max(1, int(max_children[0]))
            # Only consider positive-mass indices for sampling without replacement
            positive_mask = block_probs > 0
            pos_count = int(positive_mask.sum().item())
            if pos_count <= 0:
                block_probs = _sanitize_probs(block_probs)
                positive_mask = block_probs > 0
                pos_count = int(positive_mask.sum().item())
            Kb = min(Kb, max(1, pos_count))
            block_top_idx = torch.multinomial(block_probs, num_samples=Kb, replacement=False).detach().cpu().tolist()
        else:
            block_top_idx = sample_topk_indices(block_probs, max_children[0])

        children: List[Dict[str, Any]] = []
        for b_idx in block_top_idx:
            block_smi = idx_to_block[b_idx] if 0 <= b_idx < len(idx_to_block) else None
            if block_smi is None:
                continue
            # Reaction distribution conditional on block choice
            b_idx_tensor = torch.tensor([b_idx], dtype=torch.long, device=device)
            rxn_input = torch.cat([h_state, block_embs.index_select(0, b_idx_tensor)], dim=1)
            rxn_logits = model.reaction_head(rxn_input).squeeze(0)
            rxn_probs = softmax_logits(rxn_logits, temperature=args.temperature)
            if args.sampling_method == "nucleus":
                rxn_probs = _nucleus_filter(rxn_probs, args.nucleus_p)
            rxn_probs = _sanitize_probs(rxn_probs)
            # Feasible-set mask for rxns given chosen block at this state
            allowed_r_idxs: set[int] = set()
            for e in edges:
                if e.block_smiles == block_smi:
                    ri = rxn_to_idx.get(e.rxn_template)
                    if ri is not None:
                        allowed_r_idxs.add(int(ri))
            if allowed_r_idxs:
                maskr = torch.zeros_like(rxn_probs)
                for ri in allowed_r_idxs:
                    if 0 <= ri < maskr.numel():
                        maskr[ri] = 1.0
                rxn_probs = rxn_probs * maskr
                rxn_probs = _sanitize_probs(rxn_probs)
            sumr = float(torch.sum(rxn_probs).item())
            if sumr > 0:
                rxn_probs = rxn_probs / sumr
            else:
                # Fallback: uniform over feasible rxns if available, else uniform over all
                if allowed_r_idxs:
                    rxn_probs = torch.zeros_like(rxn_probs)
                    for ri in allowed_r_idxs:
                        if 0 <= ri < rxn_probs.numel():
                            rxn_probs[ri] = 1.0
                    rxn_probs = rxn_probs / float(torch.sum(rxn_probs).item())
                else:
                    rxn_probs = torch.ones_like(rxn_probs) / float(rxn_probs.numel())
            rxn_probs = _sanitize_probs(rxn_probs)
            if not bool(getattr(args, "deterministic", False)):
                Kr = max(1, int(max_children[1]))
                positive_mask_r = rxn_probs > 0
                pos_count_r = int(positive_mask_r.sum().item())
                if pos_count_r <= 0:
                    rxn_probs = _sanitize_probs(rxn_probs)
                    positive_mask_r = rxn_probs > 0
                    pos_count_r = int(positive_mask_r.sum().item())
                Kr = min(Kr, max(1, pos_count_r))
                rxn_top_idx = torch.multinomial(rxn_probs, num_samples=Kr, replacement=False).detach().cpu().tolist()
            else:
                rxn_top_idx = sample_topk_indices(rxn_probs, max_children[1])

            # For each top reaction, only keep those that exist as feasible edges in dataset
            for r_idx in rxn_top_idx:
                rxn_tpl = idx_to_rxn[r_idx] if 0 <= r_idx < len(idx_to_rxn) else None
                if rxn_tpl is None:
                    continue
                # Find a matching edge in the dataset graph
                viable = [e for e in edges if e.block_smiles == block_smi and e.rxn_template == rxn_tpl]
                if not viable:
                    continue
                # Deduplicate by result
                seen_res: set[str] = set()
                for e in viable:
                    if e.result_smiles in seen_res:
                        continue
                    seen_res.add(e.result_smiles)
                    # RDKit validity check to avoid invalid valence products
                    if not _is_valid_smiles(e.result_smiles):
                        continue
                    # Combine policy prob and optional QSAR score for ordering
                    policy_prob = float(block_probs[b_idx].item()) * float(rxn_probs[r_idx].item())
                    qsar_score = None
                    if qsar_predictor is not None:
                        try:
                            qsar_val = qsar_predictor.predict_pactivity(e.result_smiles, args.protein)
                            # Map to (0,1) via sigmoid for combination
                            qsar_score = 1.0 / (1.0 + float(torch.exp(torch.tensor(-qsar_val)).item()))
                        except Exception:
                            qsar_score = None
                    if qsar_predictor is not None and qsar_score is not None:
                        mix = max(0.0, min(1.0, float(args.qsar_mix)))
                        # Geometric mixture in [0,1]
                        combined = (policy_prob ** (1.0 - mix)) * (qsar_score ** mix)
                    else:
                        combined = policy_prob
                    children.append({
                        "block_smiles": e.block_smiles,
                        "rxn_template": e.rxn_template,
                        "result_smiles": e.result_smiles,
                        "next_state": e.result_smiles,
                        "policy_prob": policy_prob,
                        "qsar_score": qsar_score,
                        "combined_score": combined,
                    })
        # Last-resort fallback: if still empty but edges exist, add feasible edges directly (limited by K)
        if not children and edges:
            seen_res: set[str] = set()
            Kb = max(1, int(max_children[0]))
            Kr = max(1, int(max_children[1]))
            # Group edges by block to enforce both caps
            block_groups: Dict[str, List[Edge]] = {}
            for e in edges:
                block_groups.setdefault(e.block_smiles, []).append(e)
            b_keys = list(block_groups.keys())[:Kb]
            for b_smi in b_keys:
                sub = block_groups[b_smi][:Kr]
                for e in sub:
                    if e.result_smiles in seen_res:
                        continue
                    seen_res.add(e.result_smiles)
                    if not _is_valid_smiles(e.result_smiles):
                        continue
                    children.append({
                        "block_smiles": e.block_smiles,
                        "rxn_template": e.rxn_template,
                        "result_smiles": e.result_smiles,
                        "next_state": e.result_smiles,
                    })
        # Open-space inference: optionally add template-derived proposals
        if bool(getattr(args, "template_walk", False)):
            open_children: List[Dict[str, Any]] = []
            # Lazy init holders on closure
            nonlocal template_lib, extra_blocks
            # Load templates
            if template_lib is None and args.template_csv:
                try:
                    template_lib = TemplateLibrary.from_csv(
                        args.template_csv, max_rows=int(args.template_max_rows)
                    )
                except Exception:
                    template_lib = None
            # Load external blocks (read only the needed column and rows to speed up)
            if not extra_blocks and args.extra_blocks_csv:
                _ensure_block_library_loaded()
            # Propose via templates (with metadata so we can expose block + template in JSON)
            steps: List[Any] = []
            if state in open_cache:
                steps = open_cache[state]
            elif template_lib is not None:
                try:
                    # First try single-reactant templates
                    steps = template_lib.propose_steps(
                        state_smiles=state,
                        block_smiles=None,
                        try_limit=int(args.template_try_templates),
                        max_products_per_template=1,
                    )
                except Exception:
                    steps = []
                if not steps and extra_blocks:
                    try:
                        # Fallback: two-reactant templates with external block pool
                        steps = template_lib.propose_steps_with_pool(
                            state_smiles=state,
                            block_pool=extra_blocks,
                            try_limit_templates=int(args.template_try_templates),
                            sample_blocks=int(args.template_sample_blocks),
                            max_products_per_template=1,
                        )
                    except Exception:
                        steps = []
                open_cache[state] = steps
            # Feasibility filtering (RDKit validity) on resulting products
            if steps and str(getattr(args, 'feasibility_filter', 'rdkit')) != 'none':
                valid_steps: List[Any] = []
                for st in steps:
                    prod = getattr(st, "product_smiles", None)
                    if prod is None:
                        if isinstance(st, dict):
                            prod = st.get("product_smiles") or st.get("result_smiles") or st.get("smiles")
                        else:
                            prod = str(st)
                    prod = str(prod or "").strip()
                    if not prod:
                        continue
                    try:
                        m = Chem.MolFromSmiles(prod)
                    except Exception:
                        m = None
                    if m is not None:
                        valid_steps.append(st)
                steps = valid_steps
            # Attach proposals from template-walk
            for st in steps[: int(getattr(args, 'open_max_proposals', 8))]:
                prod = getattr(st, "product_smiles", None)
                blk = getattr(st, "block_smiles", None)
                tpl_label = getattr(st, "template_label", None)
                tpl_smarts = getattr(st, "template_smarts", None)
                if prod is None or (blk is None and tpl_label is None and tpl_smarts is None):
                    if isinstance(st, dict):
                        prod = prod or st.get("product_smiles") or st.get("result_smiles") or st.get("smiles")
                        blk = blk or st.get("block_smiles")
                        tpl_label = tpl_label or st.get("template_label")
                        tpl_smarts = tpl_smarts or st.get("template_smarts")
                prod = str(prod or "").strip()
                if not prod:
                    continue
                open_children.append({
                    "block_smiles": str(blk or ""),
                    "rxn_template": (tpl_label or tpl_smarts),
                    "result_smiles": prod,
                    "next_state": prod,
                    "policy_prob": 0.0,
                    "qsar_score": None,
                    "combined_score": 0.0,
                    "source": "template_walk",
                })
            if open_children:
                children.extend(open_children)
                # Recompute combined ordering if QSAR enabled (batch score for speed)
                if qsar_predictor is not None:
                    try:
                        smi_list = [c["result_smiles"] for c in children]
                        raw_scores = qsar_predictor.predict_pactivity_batch(smi_list, args.protein, batch_size=512)
                        for ch, rv in zip(children, raw_scores):
                            sig = 1.0 / (1.0 + float(torch.exp(torch.tensor(-rv)).item()))
                            ch["qsar_score"] = sig
                            mix = max(0.0, min(1.0, float(args.qsar_mix)))
                            pol = float(ch.get("policy_prob", 0.0))
                            ch["combined_score"] = (pol ** (1.0 - mix)) * (sig ** mix)
                    except Exception:
                        # Fallback to per-item if batch fails
                        for ch in children:
                            if ch.get("qsar_score") is None:
                                try:
                                    val = qsar_predictor.predict_pactivity(ch["result_smiles"], args.protein)
                                    ch["qsar_score"] = 1.0 / (1.0 + float(torch.exp(torch.tensor(-val)).item()))
                                except Exception:
                                    ch["qsar_score"] = None
                            mix = max(0.0, min(1.0, float(args.qsar_mix)))
                            pol = float(ch.get("policy_prob", 0.0))
                            qs = float(ch.get("qsar_score", 0.0) or 0.0)
                            ch["combined_score"] = (pol ** (1.0 - mix)) * (qs ** mix)
                    children.sort(key=lambda d: float(d.get("combined_score", 0.0)), reverse=True)
        # Guard: no-immediate-undo (filter children with heavy atom decrease)
        if bool(getattr(args, "no_immediate_undo", True)) and children:
            hvy = _heavy_atoms(state)
            if hvy >= 0:
                children = [c for c in children if _heavy_atoms(c.get("result_smiles", "")) >= hvy]

        # Additional physicalization for template-walk proposals:
        # - 单反应物 / template_walk 提案必须真正“变大”（默认至少 +1 个重原子）；
        # - 同时避免一步跳得过大（默认最多 +40 个重原子），以防止不理想的“巨跳跃模板”。
        if bool(getattr(args, "template_walk", False)) and bool(getattr(args, "template_physical_filter", True)) and children:
            hvy = _heavy_atoms(state)
            if hvy >= 0:
                min_delta = int(getattr(args, "template_min_delta_heavy", 1))
                max_delta = int(getattr(args, "template_max_delta_heavy", 40))

                def _keep_child(ch: Dict[str, Any]) -> bool:
                    res = ch.get("result_smiles", "")
                    hvy_res = _heavy_atoms(res)
                    if hvy_res < 0:
                        return False
                    delta = hvy_res - hvy
                    # 只对 open-space template_walk 的步骤施加“严格物理化”约束；
                    # 数据集里的 edge 保留原样（只受 no_immediate_undo 影响）。
                    if str(ch.get("source", "")) == "template_walk":
                        if min_delta > 0 and delta < min_delta:
                            return False
                        if max_delta > 0 and delta > max_delta:
                            return False
                    return True

                children = [c for c in children if _keep_child(c)]

        # Global cap for single-reactant steps (数据集 + template-walk 均适用)：
        # - 对于没有外部 block_smiles 的步骤（单反应物模板），如果产物重原子数比当前状态多得“离谱”，则直接过滤掉，
        #   用于抑制像 CN1CCC[C@@H]1CCCO 一步长出完整芳香磺酰胺这类“巨跳跃”模板。
        if children:
            hvy = _heavy_atoms(state)
            max_single = int(getattr(args, "single_react_max_delta_heavy", 40))
            if hvy >= 0 and max_single > 0:
                def _keep_single(ch: Dict[str, Any]) -> bool:
                    # 只针对单反应物：没有显式外部砌块的步骤
                    block_smi = str(ch.get("block_smiles", "") or "")
                    if block_smi:
                        return True
                    hvy_res = _heavy_atoms(ch.get("result_smiles", ""))
                    if hvy_res < 0:
                        return False
                    delta = hvy_res - hvy
                    # no_immediate_undo 已经保证 delta >= 0，这里只限制上限
                    return delta <= max_single

                children = [c for c in children if _keep_single(c)]

        # Optional per-step PLANTAIN re-ranking
        if bool(getattr(args, "step_plantain", True)) and bool(getattr(args, "use_plantain", False)) and str(getattr(args, "plantain_pocket", "")) and children:
            step_counter["n"] += 1
            interval = int(max(1, getattr(args, "step_plantain_interval", 1)))
            if (step_counter["n"] % interval) == 0:
                try:
                    from LeadGFlowNet.oracle import _plantain_min_score_for_smiles as _pmin, transform_plantain as _tp
                    M = min(int(getattr(args, "step_plantain_topk", 8)), len(children))
                    # Evaluate top-M by current combined_score/policy_prob ordering
                    head = children[:M]
                    rest = children[M:]
                    for ch in head:
                        smi = ch.get("result_smiles", "")
                        ps = _pmin(smi, str(getattr(args, "plantain_pocket", "")), device=str(getattr(args, "plantain_device", "auto")))
                        if ps is not None:
                            r = _tp(ps, scale=float(getattr(args, "plantain_scale", 10.0)))
                            pol = float(ch.get("combined_score", ch.get("policy_prob", 0.0)) or 0.0)
                            mix = max(0.0, min(1.0, float(getattr(args, "step_plantain_mix", 0.3))))
                            ch["combined_score"] = (pol ** (1.0 - mix)) * (float(r) ** mix)
                    head.sort(key=lambda d: float(d.get("combined_score", d.get("policy_prob", 0.0)) or 0.0), reverse=True)
                    children = head + rest
                except Exception:
                    pass

        # If QSAR enabled, sort by combined score desc
        if qsar_predictor is not None and children:
            children.sort(key=lambda d: float(d.get("combined_score", 0.0)), reverse=True)
        return children

    def sample_tree(root_state: str, max_depth: int, branch_k: Tuple[int, int]) -> Dict[str, Any]:
        if getattr(args, "expand_mode", "path") == "path":
            # Sample a single main branch, but真正按“链式”结构组织树：
            # root_state -> step1(result1) -> step2(result2) -> ...

            def _sample_path(state: str, depth: int) -> Dict[str, Any]:
                if depth >= max_depth:
                    return {"state": state, "children": []}
                childs = expand_state(state, branch_k)
                if not childs:
                    return {"state": state, "children": []}
                # pick the first child (already ordered by combined score); stochastic: sample from head window if not deterministic
                if not bool(getattr(args, "deterministic", False)) and len(childs) > 1:
                    pick = random.choice(childs[: min(3, len(childs))])
                else:
                    pick = childs[0]
                subtree = _sample_path(pick["next_state"], depth + 1)
                return {
                    "state": state,
                    "children": [
                        {
                            "block_smiles": pick["block_smiles"],
                            "rxn_template": pick["rxn_template"],
                            "result_smiles": pick["result_smiles"],
                            "subtree": subtree,
                        }
                    ],
                }

            return _sample_path(root_state, 0)
        # DFS expansion up to max_depth (full tree)
        def _expand(state: str, depth: int) -> Dict[str, Any]:
            if depth >= max_depth:
                return {"state": state, "children": []}
            childs = expand_state(state, branch_k)
            return {
                "state": state,
                "children": [
                    {
                        "block_smiles": c["block_smiles"],
                        "rxn_template": c["rxn_template"],
                        "result_smiles": c["result_smiles"],
                        "subtree": _expand(c["next_state"], depth + 1),
                    }
                    for c in childs
                ],
            }

        return _expand(root_state, 0)

    def _collect_paths(tree: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Collect all edge paths from root state to leaf result.
        Each path is a list of dicts with keys: state, block_smiles, rxn_template, result_smiles.
        """
        paths: List[List[Dict[str, Any]]] = []
        def _dfs(node: Dict[str, Any], acc: List[Dict[str, Any]]) -> None:
            children = node.get("children", [])
            if not children:
                if acc:
                    paths.append(list(acc))
                return
            for edge in children:
                step = {
                    "state": str(node.get("state", "")),
                    "block_smiles": str(edge.get("block_smiles", "")),
                    "rxn_template": str(edge.get("rxn_template", "")),
                    "result_smiles": str(edge.get("result_smiles", "")),
                }
                acc.append(step)
                subtree = edge.get("subtree", {}) or {}
                if subtree:
                    _dfs(subtree, acc)
                else:
                    paths.append(list(acc))
                acc.pop()
        _dfs(tree, [])
        return paths

    def _block_meta_payload(smiles: str) -> Optional[Dict[str, Optional[str]]]:
        smi = str(smiles or "").strip()
        if not smi:
            return None
        meta = block_library_meta.get(smi)
        if meta is not None:
            return meta
        return {
            "smiles": smi,
            "id": None,
            "size": None,
            "price": None,
        }

    def _render_forward_children(node: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not node:
            return []
        items: List[Dict[str, Any]] = []
        children = node.get("children", []) or []
        for edge in children:
            entry = {
                "current_state_smiles": node.get("state", ""),
                "building_block": _block_meta_payload(edge.get("block_smiles", "")),
                "reaction_template": edge.get("rxn_template", ""),
                "intermediate_smiles": edge.get("result_smiles", ""),
                "children": _render_forward_children(edge.get("subtree", {}) or {}),
            }
            items.append(entry)
        return items

    def _render_forward_tree(node: Dict[str, Any]) -> Dict[str, Any]:
        """Render synthesis_tree under Plan B semantics.

        根节点语义（Plan B）：
        - 起始 state_smiles_0 本身就是来自 building block 库的单一砌块；
        - JSON 中用一个“虚拟根”节点承载它：
          * current_state_smiles = null
          * building_block = state_smiles_0 在库中的信息
          * intermediate_smiles = state_smiles_0

        后续每一步沿用 state-based 表达：
        - current_state_smiles = 上一节点的 intermediate_smiles（即 node['state']）
        - building_block = 此步引入的外部砌块（单反应物模板则为 null）
        - reaction_template / intermediate_smiles = 来自 forward 轨迹的动作与结果
        """
        root_state = str(node.get("state", "") or "")
        return {
            "current_state_smiles": None,
            "building_block": _block_meta_payload(root_state),
            "reaction_template": None,
            "intermediate_smiles": root_state,
            "children": _render_forward_children(node),
        }

    # Main sampling loop (seed if provided)
    if int(getattr(args, "seed", -1)) >= 0:
        random.seed(int(args.seed))
        try:
            torch.manual_seed(int(args.seed))
        except Exception:
            pass
    routes_forward: List[Dict[str, Any]] = []
    leads_set: List[str] = []
    for i in range(max(1, args.num_samples)):
        root = random.choice(start_states)
        tree = sample_tree(root_state=root, max_depth=int(args.max_depth), branch_k=(int(args.branch_block_topk), int(args.branch_rxn_topk)))

        # Collect leaves' result_smiles
        def _collect_leads(t: Dict[str, Any]):
            # Leaf when no children
            if not t.get("children"):
                return
            for edge in t["children"]:
                subtree = edge.get("subtree", {})
                if not subtree.get("children"):
                    # Treat the edge's result as a final product
                    smi = str(edge.get("result_smiles", ""))
                    if smi and smi not in leads_set:
                        leads_set.append(smi)
                else:
                    _collect_leads(subtree)

        _collect_leads(tree)
        # Append forward tree (pre-reversed format): one route per sample
        routes_forward.append({
            "root_smiles": root,
            "synthesis_tree": _render_forward_tree(tree),
        })
        # Progress indicator file (append 1-based index)
        pf = str(getattr(args, "progress_file", "")).strip()
        if pf:
            try:
                os.makedirs(os.path.dirname(pf) or ".", exist_ok=True)
            except Exception:
                pass
            try:
                with open(pf, "a", encoding="utf-8") as pff:
                    pff.write(f"{i+1}\n")
            except Exception:
                pass

    # Post-selection: build a diverse subset of leads
    def _score_smiles(smi: str) -> float:
        if bool(args.use_qsar) and qsar_predictor is not None:
            try:
                val = float(qsar_predictor.predict_pactivity(smi, args.protein))
                return 1.0 / (1.0 + float(torch.exp(torch.tensor(-val)).item()))
            except Exception:
                pass
        try:
            m = Chem.MolFromSmiles(smi)
            return float(QED.qed(m)) if m is not None else 0.0
        except Exception:
            return 0.0

    def _select_diverse(candidates: List[str], k: int) -> List[str]:
        k = max(1, int(k))
        if args.diversity_mode == "none":
            return candidates[:k]
        items = []
        for s in candidates:
            fp = _fp_ecfp4(s)
            sc = _score_smiles(s)
            items.append((s, fp, sc))
        selected: List[str] = []
        if args.diversity_mode == "minsim":
            th = float(args.minsim_th)
            for s, fp, sc in items:
                if fp is None:
                    continue
                ok = True
                for t in selected:
                    fp_t = _fp_ecfp4(t)
                    if fp_t is None:
                        continue
                    if _tanimoto(fp, fp_t) > th:
                        ok = False
                        break
                if ok:
                    selected.append(s)
                if len(selected) >= k:
                    break
            return selected
        # MMR
        lam = max(0.0, min(1.0, float(args.mmr_lambda)))
        picked = []
        pool = items[:]
        while pool and len(selected) < k:
            best = None
            best_val = -1e9
            for s, fp, sc in pool:
                if not picked:
                    val = sc
                else:
                    max_sim = 0.0
                    if fp is not None:
                        for s2, fp2, sc2 in picked:
                            if fp2 is None:
                                continue
                            max_sim = max(max_sim, _tanimoto(fp, fp2))
                    val = lam * sc - (1.0 - lam) * max_sim
                if val > best_val:
                    best_val = val
                    best = (s, fp, sc)
            if best is None:
                break
            picked.append(best)
            selected.append(best[0])
            pool.remove(best)
        return selected

    leads_diverse = _select_diverse(leads_set, int(getattr(args, "select_k", 200)))

    # Optional: QSAR-ranked export over all unique leads
    ranked_info = None
    saved_ranked_csv = None
    saved_ranked_json = None
    saved_tree_csv = None
    if bool(getattr(args, "export_ranked", False)):
        # Optionally build QSAR predictor even if --use-qsar not set
        if qsar_predictor is None and os.path.exists(args.qsar_checkpoint):
            try:
                qsar_predictor = QSARPredictor(args.qsar_checkpoint, device=device)
            except Exception:
                qsar_predictor = None
        # Optional PLANTAIN scorer + Vina refine
        def _plantain_min(smi: str) -> Optional[float]:
            if not use_plantain or not plantain_pocket:
                return None
            try:
                from LeadGFlowNet import oracle as _oracle
                val = _oracle._plantain_min_score_for_smiles(smi, plantain_pocket, device=str(getattr(args, "plantain_device", "auto")))
                return float(val) if val is not None else None
            except Exception:
                return None

        # Helpers for Vina refine
        def _bbox_from_pdb_like(path: str):
            xs = []; ys = []; zs = []
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for ln in f:
                        if ln.startswith("ATOM") or ln.startswith("HETATM"):
                            try:
                                x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
                            except Exception:
                                parts = ln.split()
                                if len(parts) < 9:
                                    continue
                                x = float(parts[-6]); y = float(parts[-5]); z = float(parts[-4])
                            xs.append(x); ys.append(y); zs.append(z)
                return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))
            except Exception:
                return ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

        def _prepare_receptor(pocket_pdb: str) -> str:
            rec_fix = pocket_pdb.replace(".pdb", "_rigid.pdbqt")
            # Try ADFR/ADT
            for prep_cmd in ("prepare_receptor", "prepare_receptor4.py"):
                try:
                    r = __import__("subprocess").run([prep_cmd, "-r", pocket_pdb, "-o", rec_fix, "-A", "checkhydrogens"], stdout=__import__("subprocess").PIPE, stderr=__import__("subprocess").STDOUT, text=True)
                    if r.returncode == 0 and os.path.exists(rec_fix) and os.path.getsize(rec_fix) > 0:
                        print({"vina_receptor": {"method": prep_cmd, "path": rec_fix}})
                        return rec_fix
                except FileNotFoundError:
                    continue
                except Exception:
                    continue
            # Require ADFRtools for receptor preparation; no OpenBabel fallback for receptors
            raise RuntimeError("ADFR prepare_receptor not available; ensure ADFRtools is installed and 'prepare_receptor' is in PATH")

        def _meeko_pdbqt_from_smiles_with_plantain(smi: str, top_k: int, tmp_dir: str, poses_dir: str) -> Optional[str]:
            # Run PLANTAIN to get pose; write top-1 SDF (ABS path); then Meeko to PDBQT (ABS path). Avoid CWD issues.
            try:
                plant_dir = os.path.join(os.path.dirname(__file__), "lib", "plantain")
                plant_dir = os.path.abspath(plant_dir)
                if plant_dir not in sys.path:
                    sys.path.insert(0, plant_dir)
                from common.cfg_utils import get_config  # type: ignore
                from models.pretrained_plantain import get_pretrained_plantain  # type: ignore
                from datasets.inference_dataset import InferenceDataset  # type: ignore
                from terrace import collate  # type: ignore
                from rdkit import Chem  # type: ignore
                from common.pose_transform import add_multi_pose_to_mol  # type: ignore
                # Normalize all working paths to ABSOLUTE before chdir
                poses_dir_abs = os.path.abspath(poses_dir)
                pdbqt_dir_abs = os.path.abspath(str(getattr(args, "vina_pdbqt_dir", "runs/vina_pdbqt")))
                tmp_dir_abs = os.path.abspath(tmp_dir)
                # Ensure Plantain is run under its repo cwd so configs/ resolve
                _old = os.getcwd()
                try:
                    os.chdir(plantain_dir := plant_dir)
                    # Explicitly point to configs folder to avoid cwd issues
                    cfg = get_config("icml", folder=os.path.join(plantain_dir, "configs"))
                    # Disable torch.compile for compatibility
                    try:
                        cfg.platform["compile"] = False  # type: ignore[index]
                    except Exception:
                        try:
                            setattr(cfg.platform, "compile", False)
                        except Exception:
                            pass
                    model = get_pretrained_plantain()
                    try:
                        model.eval()
                    except Exception:
                        pass
                    smi_path = os.path.join(tmp_dir_abs, "one.smi"); os.makedirs(tmp_dir_abs, exist_ok=True)
                    open(smi_path, "w", encoding="utf-8").write(smi + "\n")
                    ds = InferenceDataset(cfg, smi_path, plantain_pocket, model.get_input_feats())
                    if len(ds) <= 0:
                        print({"plantain_error": "empty_dataset", "smi": smi[:32], "pocket": plantain_pocket})
                        return None
                    x, y = ds[0]
                    batch = collate([x])
                    try:
                        dev = str(getattr(args, "plantain_device", "auto")).lower()
                        if dev == "auto":
                            try:
                                if torch.cuda.is_available():
                                    dev = "cuda:0"
                                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                                    dev = "mps"
                                else:
                                    dev = "cpu"
                            except Exception:
                                dev = "cpu"
                        batch = batch.to(dev)
                        model = model.to(dev)
                    except Exception:
                        pass
                    try:
                        pred = model(batch)[0]
                    except Exception as e_inf:
                        print({"plantain_infer_error": str(e_inf)})
                        return None
                    mol = getattr(x, "lig", None)
                    if mol is None or not hasattr(pred, "lig_pose") or pred.lig_pose is None:
                        print({"plantain_error": "no_lig_pose"})
                        return None
                    add_multi_pose_to_mol(mol, pred.lig_pose)
                    # Save pose SDF to the configured poses_dir (ABS) with a stable name
                    os.makedirs(poses_dir_abs, exist_ok=True)
                    sdf_path = os.path.join(poses_dir_abs, f"{abs(hash(smi))}.sdf")
                    w = Chem.SDWriter(sdf_path)
                    try:
                        # write top-1 conformer only for speed
                        w.write(mol, confId=0)
                    finally:
                        w.close()
                    print({"plantain_pose_sdf": sdf_path})
                finally:
                    try:
                        os.chdir(_old)
                    except Exception:
                        pass
                # Meeko convert
                try:
                    from meeko import MoleculePreparation, PDBQTWriterLegacy  # type: ignore
                    # Convert the saved poses_dir SDF
                    from rdkit import Chem  # type: ignore
                    mol2 = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)[0]
                    try:
                        mol2 = Chem.AddHs(mol2, addCoords=True)
                    except Exception:
                        mol2 = Chem.AddHs(mol2)
                    prep = MoleculePreparation(); u = prep.prepare(mol2)
                    if isinstance(u, (list, tuple)):
                        u = u[0]
                    s = PDBQTWriterLegacy().write_string(u, bad_charge_ok=True)
                    if isinstance(s, tuple):
                        s = s[0]
                    if not s or not str(s).strip():
                        print({"meeko_pdbqt_empty_string": True, "smiles": smi[:32]})
                        return None
                    lig_pdbqt = os.path.join(pdbqt_dir_abs, f"{abs(hash(smi))}.pdbqt")
                    os.makedirs(os.path.dirname(lig_pdbqt), exist_ok=True)
                    with open(lig_pdbqt, "w", encoding="utf-8") as _pf:
                        _pf.write(s)
                    try:
                        if os.path.getsize(lig_pdbqt) <= 0:
                            print({"meeko_pdbqt_empty_file": lig_pdbqt})
                            return None
                    except Exception:
                        return None
                    print({"meeko_pdbqt": lig_pdbqt})
                    return lig_pdbqt
                except Exception:
                    # fallback obabel if allowed
                    if bool(getattr(args, "vina_strict", False)):
                        return None
                    obabel_bin = str(getattr(args, "vina_obabel_bin", "/usr/local/bin/obabel"))
                    lig_pdbqt = os.path.join(pdbqt_dir_abs, f"{abs(hash(smi))}.pdbqt")
                    __import__("subprocess").run([obabel_bin, "-isdf", sdf_path, "-opdbqt", "-O", lig_pdbqt, "-f", "1", "-l", "1", "-h"], check=True)
                    print({"meeko_fallback_obabel_pdbqt": lig_pdbqt})
                    return lig_pdbqt if os.path.exists(lig_pdbqt) else None
                except Exception as e_outer:
                    print({"plantain_outer_error": str(e_outer)})
                    # Final fallback: build 3D from SMILES and convert via obabel
                    if bool(getattr(args, "vina_strict", False)):
                        return None
                    try:
                        m = Chem.MolFromSmiles(smi)
                        if m is None:
                            return None
                        m = Chem.AddHs(m)
                        AllChem.EmbedMolecule(m, AllChem.ETKDG())
                        import tempfile as _tmp
                        td2 = _tmp.mkdtemp(prefix="infer_vina_pose_")
                        # Also save fallback SDF in poses_dir for visibility
                        os.makedirs(poses_dir_abs, exist_ok=True)
                        sdf2 = os.path.join(poses_dir_abs, f"{abs(hash(smi))}_fallback.sdf")
                        from rdkit.Chem import SDWriter
                        SDWriter(sdf2).write(m)
                        obabel_bin = str(getattr(args, "vina_obabel_bin", "/usr/local/bin/obabel"))
                        lig_pdbqt = os.path.join(pdbqt_dir_abs, f"{abs(hash(smi))}.pdbqt")
                        os.makedirs(os.path.dirname(lig_pdbqt), exist_ok=True)
                        __import__("subprocess").run([obabel_bin, "-isdf", sdf2, "-opdbqt", "-O", lig_pdbqt, "-h"], check=True)
                        print({"meeko_fallback_smiles3d_pdbqt": lig_pdbqt})
                        return lig_pdbqt if os.path.exists(lig_pdbqt) else None
                    except Exception:
                        return None

            except Exception as e_all:
                print({"plantain_total_error": str(e_all)})
                return None

        # Prepare PAINS filter once
        pains_catalog = None
        try:
            params = FilterCatalog.FilterCatalogParams()
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
            pains_catalog = FilterCatalog.FilterCatalog(params)
        except Exception:
            pains_catalog = None

        ranked: List[Dict[str, Any]] = []
        # Initialize shared Vina instance/grid once
        vina_ready = False
        vina_obj = None
        grid_center = None
        grid_size = None
        receptor_pdbqt = None
        if bool(getattr(args, "use_vina", False)) and plantain_pocket:
            try:
                receptor_pdbqt = _prepare_receptor(plantain_pocket)
                if str(getattr(args, "vina_center", "")):
                    grid_center = [float(x) for x in str(getattr(args, "vina_center", "")).split(",")]
                else:
                    (px, py, pz) = _bbox_from_pdb_like(plantain_pocket)
                    grid_center = [(px[0]+px[1])/2, (py[0]+py[1])/2, (pz[0]+pz[1])/2]
                s = float(getattr(args, "vina_box_size", 22.0))
                grid_size = [max(16.0, min(60.0, s))]*3
                from vina import Vina  # type: ignore
                vina_obj = Vina(sf_name="vina")
                vina_obj.set_receptor(receptor_pdbqt)
                vina_obj.compute_vina_maps(center=grid_center, box_size=grid_size)
                vina_ready = True
                print({"vina_grid": {"center": grid_center, "size": grid_size}})
            except Exception:
                vina_ready = False
        QED_MIN_TH = 0.3
        import tempfile as _tmp
        tmp_root = _tmp.mkdtemp(prefix="plantain_pose_") if bool(getattr(args, "use_vina", False)) else None
        # 只对去重+多样性筛选后的 leads_diverse 做 Plantain+Vina 精排；
        # 若由于参数设置导致 leads_diverse 为空，则回退到对全部 leads_set 进行评估。
        dock_candidates = leads_diverse if leads_diverse else leads_set
        for smi in dock_candidates:
            # Validate and canonicalize SMILES early to avoid truncated/invalid prints
            canon_smi: Optional[str] = None
            m0 = None
            try:
                m0 = Chem.MolFromSmiles(smi)
                if m0 is not None:
                    try:
                        canon_smi = Chem.MolToSmiles(m0, canonical=True)
                    except Exception:
                        canon_smi = None
            except Exception:
                m0 = None
            rec: Dict[str, Any] = {"smiles": (canon_smi or smi)}
            # QSAR
            if qsar_predictor is not None:
                try:
                    val = float(qsar_predictor.predict_pactivity(smi, args.protein))
                    sig = 1.0 / (1.0 + float(torch.exp(torch.tensor(-val)).item()))
                    rec["qsar_raw"] = val
                    rec["qsar_sigmoid"] = sig
                except Exception:
                    pass
            # QED
            m = m0
            if m is None:
                try:
                    m = Chem.MolFromSmiles(smi)
                except Exception:
                    m = None
            try:
                qv = float(QED.qed(m)) if m is not None else 0.0
            except Exception:
                qv = 0.0
            # Filter by QED threshold
            if qv < QED_MIN_TH:
                continue
            # Filter PAINS-like tox
            try:
                if pains_catalog is not None and m is not None and pains_catalog.HasMatch(m):
                    continue
            except Exception:
                pass
            rec["qed"] = qv
            # PLANTAIN
            pm = _plantain_min(smi)
            if pm is not None:
                rec["plantain_min"] = pm
                # Calibrate by heavy-atom count: new_score = old_score / sqrt(n_heavy)
                try:
                    m_hvy = 0
                    if m is None:
                        try:
                            m = Chem.MolFromSmiles(smi)
                        except Exception:
                            m = None
                    if m is not None:
                        m_hvy = int(m.GetNumHeavyAtoms())
                    denom = (m_hvy if m_hvy and m_hvy > 0 else 1) ** 0.5
                    rec["new_score"] = float(pm) / float(denom)
                except Exception:
                    rec["new_score"] = None
                # Keep legacy reward column (not used for sorting anymore)
                try:
                    from LeadGFlowNet.oracle import transform_plantain as _tp
                    rec["plantain_reward"] = float(_tp(pm, scale=10.0))
                except Exception:
                    rec["plantain_reward"] = None
            # Vina refine
            if bool(getattr(args, "use_vina", False)) and vina_ready and receptor_pdbqt and grid_center and grid_size:
                try:
                    # Prepare ligand from PLANTAIN pose
                    lig_pdbqt = _meeko_pdbqt_from_smiles_with_plantain(
                        smi,
                        int(getattr(args, "vina_top_k", 3)),
                        os.path.join(tmp_root or ".", str(abs(hash(smi)))),
                        str(getattr(args, "plantain_poses_dir", "runs/plantain_poses")),
                    )
                    if lig_pdbqt:
                        v = vina_obj
                        try:
                            v.set_ligand_from_file(lig_pdbqt)
                            res = v.score()
                            try:
                                raw = float(res)
                            except Exception:
                                import numpy as _np  # type: ignore
                                raw = float(_np.asarray(res).ravel()[0])
                            try:
                                v.optimize()
                            except Exception:
                                pass
                            res2 = v.score()
                            try:
                                opt = float(res2)
                            except Exception:
                                import numpy as _np2  # type: ignore
                                opt = float(_np2.asarray(res2).ravel()[0])
                            print({"vina_scores": {"smiles": rec.get("smiles", smi), "raw": raw, "min": opt}})
                            if opt is None or (opt is not None and opt > float(getattr(args, "vina_full_dock_th", -3.0))):
                                try:
                                    v.dock(exhaustiveness=int(max(8, getattr(args, "vina_exhaustiveness", 32))), n_poses=1)
                                    res3 = v.score();
                                    try:
                                        opt2 = float(res3)
                                    except Exception:
                                        import numpy as _np3  # type: ignore
                                        opt2 = float(_np3.asarray(res3).ravel()[0])
                                    if opt is None or (opt2 is not None and opt2 < opt):
                                        opt = opt2
                                    print({"vina_dock_refine": {"smiles": rec.get("smiles", smi), "min": opt}})
                                except Exception as e:
                                    print({"vina_dock_error": str(e)})
                            rec["vina_affinity_raw"] = raw
                            rec["vina_affinity"] = (opt if opt is not None else raw)
                        except Exception as e:
                            print({"vina_set_ligand_or_score_error": str(e), "ligand_pdbqt": lig_pdbqt})
                            # Recovery: try a quick dock then re-score
                            try:
                                v.dock(exhaustiveness=int(max(8, getattr(args, "vina_exhaustiveness", 32))), n_poses=1)
                                resd = v.score()
                                try:
                                    opt = float(resd)
                                except Exception:
                                    import numpy as _np4  # type: ignore
                                    opt = float(_np4.asarray(resd).ravel()[0])
                                rec["vina_affinity"] = opt
                                rec["vina_affinity_raw"] = None
                                print({"vina_recovery_dock": {"smiles": rec.get("smiles", smi), "min": opt}})
                            except Exception as e2:
                                print({"vina_recovery_failed": str(e2)})
                except Exception:
                    pass
            ranked.append(rec)

        # Default ordering: QSAR sigmoid desc; else by Vina affinity asc (lower better); else by calibrated PLANTAIN new_score desc; else by qed desc
        def _key(d: Dict[str, Any]) -> float:
            if "qsar_sigmoid" in d:
                return float(d.get("qsar_sigmoid", 0.0))
            if "vina_affinity" in d and d.get("vina_affinity") is not None and str(d.get("vina_affinity")) != "":
                # sort ascending by energy => invert sign for reverse=True
                return -float(d.get("vina_affinity", 0.0))
            if "new_score" in d:
                return float(d.get("new_score", 0.0))
            return float(d.get("qed", 0.0))
        ranked.sort(key=_key, reverse=True)

        # Threshold filter on QSAR if requested and available
        th = float(getattr(args, "min_qsar", 0.0))
        if th > 0.0:
            ranked = [r for r in ranked if float(r.get("qsar_sigmoid", 0.0)) >= th]
        ranked_info = ranked

        # Save CSV with multiple columns (optionally filtered by vina threshold)
        try:
            os.makedirs(os.path.dirname(args.output_ranked_csv) or ".", exist_ok=True)
            cols = ["smiles", "vina_affinity", "plantain_min", "new_score", "plantain_reward", "qsar_sigmoid", "qsar_raw", "qed", "vina_affinity_raw"]
            with open(args.output_ranked_csv, "w", encoding="utf-8") as f:
                f.write(",".join(cols) + "\n")
                ranked_for_csv = ranked
                try:
                    th = getattr(args, "filter_vina_th", None)
                except Exception:
                    th = None
                if th is not None:
                    try:
                        thr = float(th)
                        ranked_for_csv = [r for r in ranked if (r.get("vina_affinity") is not None and str(r.get("vina_affinity")) != "" and float(r.get("vina_affinity")) < thr)]
                    except Exception:
                        ranked_for_csv = ranked
                for r in ranked_for_csv:
                    row = [
                        str(r.get("smiles", "")),
                        ("" if r.get("vina_affinity") is None else f"{float(r['vina_affinity']):.6f}"),
                        ("" if r.get("plantain_min") is None else f"{float(r['plantain_min']):.6f}"),
                        ("" if r.get("new_score") is None else f"{float(r['new_score']):.6f}"),
                        ("" if r.get("plantain_reward") is None else f"{float(r['plantain_reward']):.6f}"),
                        ("" if r.get("qsar_sigmoid") is None else f"{float(r['qsar_sigmoid']):.6f}"),
                        ("" if r.get("qsar_raw") is None else f"{float(r['qsar_raw']):.6f}"),
                        ("" if r.get("qed") is None else f"{float(r['qed']):.6f}"),
                        ("" if r.get("vina_affinity_raw") is None else f"{float(r['vina_affinity_raw']):.6f}"),
                    ]
                    f.write(",".join(row) + "\n")
            saved_ranked_csv = args.output_ranked_csv
        except Exception:
            saved_ranked_csv = None
        # Save JSON (array of objects with only {"smiles": ...})
        try:
            os.makedirs(os.path.dirname(args.output_ranked_json) or ".", exist_ok=True)
            with open(args.output_ranked_json, "w", encoding="utf-8") as fj:
                json.dump([{"smiles": r["smiles"]} for r in ranked], fj, ensure_ascii=False, indent=2)
            saved_ranked_json = args.output_ranked_json
        except Exception:
            saved_ranked_json = None

    # Save JSON
    out_obj = {
        "protein": args.protein,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "params": {
            "checkpoint": args.checkpoint,
            "num_samples": args.num_samples,
            "max_depth": args.max_depth,
            "branch_block_topk": args.branch_block_topk,
            "branch_rxn_topk": args.branch_rxn_topk,
            "temperature": args.temperature,
            "protein_encoder": args.protein_encoder,
            "esm2_model": args.esm2_model if args.protein_encoder == "esm2" else None,
            "use_qsar": bool(args.use_qsar),
            "qsar_checkpoint": args.qsar_checkpoint if bool(args.use_qsar) else None,
            "qsar_mix": float(args.qsar_mix) if bool(args.use_qsar) else None,
            "deterministic": bool(getattr(args, "deterministic", False)),
            "seed": int(getattr(args, "seed", -1)),
            "sampling_method": args.sampling_method,
            "nucleus_p": float(args.nucleus_p) if args.sampling_method == "nucleus" else None,
            "diversity_mode": args.diversity_mode,
            "select_k": int(getattr(args, "select_k", 200)),
            "minsim_th": float(args.minsim_th) if args.diversity_mode == "minsim" else None,
            "mmr_lambda": float(args.mmr_lambda) if args.diversity_mode == "mmr" else None,
        },
        "leads_set": leads_set,
        "leads_diverse": leads_diverse,
        "routes": routes_forward,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print({
        "saved": args.output_json,
        "num_routes": len(routes_forward),
        "num_leads": len(leads_set),
        "ranked_csv": saved_ranked_csv,
        "ranked_json": saved_ranked_json,
    })


if __name__ == "__main__":
    main()


