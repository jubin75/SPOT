from __future__ import annotations

import argparse
import os
import sys
import random
from typing import Dict, List, Tuple, Optional
from collections import Counter
import json

"""
Ensure project root is on sys.path so absolute script execution can import sibling packages
"""
try:
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
except Exception:
    pass

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from SynthPolicyNet.train_policy import build_forward_dataset
from SynthPolicyNet.datasets import ForwardTrajectoryDataset
from SynthPolicyNet.models import SynthPolicyNet
from LeadGFlowNet.conditional_policy import ConditionalSynthPolicy
from LeadGFlowNet.protein_encoder import SimpleProteinEncoder, tokenize_protein
from LeadGFlowNet.trainer import LeadGFlowNetTrainer, MixedRewardController
from LeadGFlowNet.template_expander import TemplateLibrary
from rdkit import Chem  # type: ignore
from rdkit.Chem import AllChem  # type: ignore
from rdkit import DataStructs  # type: ignore
from rdkit import RDLogger  # type: ignore
from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore
from rdkit.Chem import Descriptors  # type: ignore

# Suppress RDKit warnings/errors globally to reduce noisy template mapping messages
RDLogger.DisableLog('rdApp.warning')
try:
    RDLogger.DisableLog('rdApp.error')
except Exception:
    pass


def _connect_mols_random(state_smiles: str, block_smiles: str, *, max_pair_tries: int = 64) -> Optional[str]:
    """Attempt to connect two molecules by adding a single bond between atoms
    with positive implicit valence. Returns canonical SMILES or None on failure.
    """
    try:
        m1 = Chem.MolFromSmiles(state_smiles) if state_smiles else None
        m2 = Chem.MolFromSmiles(block_smiles) if block_smiles else None
        if m1 is None or m2 is None:
            return None
        # collect candidate atoms with available valence (exclude H)
        a1 = [i for i, a in enumerate(m1.GetAtoms()) if a.GetAtomicNum() != 1 and (a.GetImplicitValence() or 0) > 0]
        a2 = [i for i, a in enumerate(m2.GetAtoms()) if a.GetAtomicNum() != 1 and (a.GetImplicitValence() or 0) > 0]
        if not a1 or not a2:
            return None
        import random as _rc
        combo = Chem.CombineMols(m1, m2)
        rw = Chem.RWMol(combo)
        off = m1.GetNumAtoms()
        for _ in range(max(1, int(max_pair_tries))):
            i = _rc.choice(a1)
            j = _rc.choice(a2)
            try:
                rw.AddBond(int(i), int(off + j), order=Chem.BondType.SINGLE)
                new = rw.GetMol()
                Chem.SanitizeMol(new)
                s = Chem.MolToSmiles(new)
                if s:
                    return s
            except Exception:
                # revert last bond and retry a different pair
                try:
                    last_bond_idx = rw.GetNumBonds() - 1
                    if last_bond_idx >= 0:
                        rw.RemoveBond(int(i), int(off + j))
                except Exception:
                    pass
                continue
    except Exception:
        return None
    return None


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Online TB fine-tuning using QSAR-only reward")
    p.add_argument("--input", default="data/reaction_paths_all_routes.csv")
    p.add_argument("--forward", default="data/forward_trajectories.csv")
    p.add_argument("--rebuild-forward", action="store_true")
    p.add_argument("--max-block-mw", type=float, default=200.0, help="Maximum molecular weight for building block reactants (Da)")
    p.add_argument("--checkpoint", default="checkpoints/synth_policy_net.pt")
    p.add_argument("--qsar-checkpoint", default="checkpoints/qsar.pt")
    # Plantain integration
    p.add_argument("--use-plantain", action="store_true", help="Use PLANTAIN as primary reward instead of QSAR")
    p.add_argument("--plantain-pocket", type=str, default="", help="Path to pocket PDB for PLANTAIN (if empty, auto-detect under test/<PDBID>/*_pocket.pdb)")
    p.add_argument("--plantain-device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"], help="Device for PLANTAIN model")
    p.add_argument("--plantain-scale", type=float, default=10.0, help="Exponential reward scale for PLANTAIN mapping (smaller => more sensitive)")
    # Vina refine (Plantain+Vina) options for reward
    p.add_argument("--use-vina", action="store_true", help="Use python-vina to refine Plantain poses for reward")
    p.add_argument("--vina-box-size", type=float, default=22.0, help="Cubic Vina grid box size (A)")
    p.add_argument("--vina-exhaustiveness", type=int, default=32, help="Docking exhaustiveness for refine")
    p.add_argument("--vina-top-k", type=int, default=1, help="Top-K Plantain poses to refine per ligand")
    p.add_argument("--vina-full-dock-th", type=float, default=-3.0, help="If optimized energy > th, run a quick dock")
    p.add_argument("--vina-obabel-bin", type=str, default="/usr/local/bin/obabel", help="Path to obabel binary")
    p.add_argument("--vina-strict", action="store_true", help="Require ADFR/Meeko; error if unavailable (no obabel fallback)")
    p.add_argument("--vina-pdbqt-dir", type=str, default="runs/vina_pdbqt", help="Where to cache ligand PDBQT files")
    p.add_argument("--vina-reward-smooth", type=float, default=0.0, help="EMA smoothing factor for Vina energy in [0,1); 0 disables")
    p.add_argument("--vina-weight", type=float, default=1.0, help="Scale factor on docking energy when mapping to reward: reward += -vina_weight * E")
    p.add_argument("--prune-bad-vina-th", type=float, default=0.0, help="If < 0: skip episodes with Vina energy > threshold (e.g., -5.0); 0 disables")
    # Pruning by molecular weight (skip overly large ligands)
    p.add_argument("--prune-mw-th", type=float, default=0.0, help="If > 0: skip episodes with molecular weight > threshold (Da); 0 disables")
    # Auto-reference collection based on Vina affinity
    p.add_argument("--auto-ref-vina-th", type=float, default=0.0, help="If < 0: when Vina affinity < threshold, append terminal SMILES to --auto-ref-out (.smi)")
    p.add_argument("--auto-ref-out", type=str, default="", help="Path to write collected reference SMILES (.smi). Default runs/ref_auto_top.smi under project root")
    p.add_argument("--auto-ref-use-scaffold", action="store_true", help="Use the collected auto-ref .smi as scaffold references during training (finetune stage)")
    p.add_argument("--auto-ref-scaffold-weight", type=float, default=0.2, help="Scaffold reward weight when auto-ref scaffold is enabled")
    # New reward controls
    p.add_argument("--use-qsar-reward", action="store_true", help="Use QSAR prediction as terminal reward")
    p.add_argument("--use-docking-guidance", action="store_true", help="Add docking-like guidance on top of QSAR (e.g., Plantain)")
    p.add_argument("--docking-model", type=str, default="plantain", choices=["plantain", "none"], help="Docking guidance backend")
    # Scaffold/fragment matching
    p.add_argument("--ref-ligands", type=str, default="", help="Path to reference ligands .smi for scaffold reward")
    p.add_argument("--use-scaffold-reward", action="store_true", help="Encourage Murcko scaffold similarity to reference ligands")
    p.add_argument("--scaffold-weight", type=float, default=0.2, help="Weight of scaffold reward added to terminal reward")
    p.add_argument("--training-stage", type=str, default="pretrain", choices=["pretrain", "finetune"], help="Stage switch for enabling scaffold reward")
    # Local/step-wise reward shaping
    p.add_argument("--use-local-reward", action="store_true", help="Enable per-step local reward/penalty for structural sanity")
    p.add_argument("--local-reward-weight", type=float, default=0.1, help="Weight for accumulated local reward added to terminal reward")
    # Per-step Plantain shaping (default on)
    p.add_argument("--perstep-plantain", dest="perstep_plantain", action="store_true", help="Enable per-step PLANTAIN shaping on states")
    p.add_argument("--perstep-plantain-interval", type=int, default=1, help="Evaluate PLANTAIN shaping every N steps (>=1)")
    p.add_argument("--perstep-plantain-weight", type=float, default=0.05, help="Weight for per-step PLANTAIN shaping added to local reward")
    p.add_argument("--pactivity", default="data/protein_ligand_pactivity.csv", help="CSV with columns ligand_smiles, protein_sequence, p_activity")
    p.add_argument("--protein-fasta", type=str, default="", help="Path to FASTA file containing protein sequence(s), or direct FASTA sequence string")
    p.add_argument("--protein-sequence", type=str, default="", help="Direct protein sequence string (alternative to --protein-fasta)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--episodes-per-epoch", type=int, default=800)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    # Reward mixing weights
    p.add_argument("--add-qed", type=float, default=0.3, help="Weight for QED reward term")
    p.add_argument("--sub-sa", type=float, default=0.05, help="Weight for SA penalty term")
    p.add_argument("--lipinski-penalty", type=float, default=0.1, help="Penalty per Lipinski violation")
    # Distributed
    p.add_argument("--distributed", type=str, default="none", choices=["none", "ddp"], help="Enable DDP")
    p.add_argument("--dist-backend", type=str, default="nccl")
    p.add_argument("--dist-init-method", type=str, default="env://")
    p.add_argument("--rxn-first", action="store_true", help="Use reaction-first factorization during sampling")
    # Model hyperparameters (0 means: infer from checkpoint if available)
    p.add_argument("--hidden-dim", type=int, default=0)
    p.add_argument("--num-gnn-layers", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--share-encoders", action="store_true")
    p.add_argument("--use-l2-norm", action="store_true")
    p.add_argument("--temperature", type=float, default=0.07)
    # Saving
    p.add_argument("--save", type=str, default="checkpoints/leadgflownet_online_tb.pt", help="Path to save online TB checkpoint each epoch")
    # Training sampling controls (diversity/feasibility)
    p.add_argument("--start-states-cap", type=int, default=256, help="Cap of distinct start states sampled per epoch")
    p.add_argument("--train-branch-block-topk", type=int, default=5, help="During training, consider top-K blocks per state")
    p.add_argument("--train-branch-rxn-topk", type=int, default=3, help="During training, consider top-K reactions per selected block")
    p.add_argument("--train-sample-temperature", type=float, default=1.0, help="Temperature for stochastic sampling at training time")
    p.add_argument("--train-deterministic", action="store_true", help="If set, pick argmax instead of stochastic sampling during training")
    p.add_argument("--teacher-forcing-prob", type=float, default=0.3, help="Per-step probability to pick a feasible (block, rxn) pair uniformly from dataset edges at the current state; log_pf still uses model probs")
    p.add_argument("--tb-residual-clip", type=float, default=10.0, help="Clip TB residual before squaring to stabilize training")
    p.add_argument("--min-start-degree", type=int, default=2, help="Minimum out-degree for start state preference")
    p.add_argument("--per-step-retries", type=int, default=2, help="Retries per step using feasible (block, rxn) pairs if initial pick fails")
    # Schedules (linear across epochs)
    p.add_argument("--tf-start", type=float, default=0.6, help="Teacher forcing prob at epoch 1")
    p.add_argument("--tf-end", type=float, default=0.2, help="Teacher forcing prob at last epoch")
    p.add_argument("--temp-start", type=float, default=1.1, help="Sampling temperature at epoch 1")
    p.add_argument("--temp-end", type=float, default=0.9, help="Sampling temperature at last epoch")
    p.add_argument("--topk-block-start", type=int, default=8, help="Block top-k at epoch 1")
    p.add_argument("--topk-block-end", type=int, default=4, help="Block top-k at last epoch")
    p.add_argument("--topk-rxn-start", type=int, default=4, help="Rxn top-k at epoch 1")
    p.add_argument("--topk-rxn-end", type=int, default=2, help="Rxn top-k at last epoch")
    # Template expansion options
    p.add_argument("--template-csv", type=str, default="", help="CSV/XLSX of reaction templates (SMARTS/SMIRKS column; auto-detect common names)")
    p.add_argument("--template-prob", type=float, default=0.0, help="Per-step probability to use template expansion instead of dataset edges")
    p.add_argument("--template-max-rows", type=int, default=5000, help="Max templates to load from CSV for speed")
    p.add_argument("--template-try-templates", type=int, default=128, help="Max number of templates to attempt per step when expanding")
    p.add_argument("--template-sample-blocks", type=int, default=2048, help="Max number of external blocks sampled per step for 2-reactant templates")
    # Pure free-connect options (no templates): directly attach external blocks via single-bond connecting
    p.add_argument("--free-connect", action="store_true", help="Bypass templates: randomly connect state to external blocks if possible")
    p.add_argument("--free-connect-prob", type=float, default=1.0, help="Per-step probability to attempt free-connect when enabled")
    p.add_argument("--free-connect-tries", type=int, default=64, help="Max random atom-pair attempts per block during free-connect")
    p.add_argument("--free-connect-sample-blocks", type=int, default=1024, help="Max number of external blocks sampled per step for free-connect")
    # Open-space training and shaping
    p.add_argument("--free-walk", action="store_true", help="Allow template-expanded products as next states even if not in dataset graph")
    p.add_argument("--template-walk", action="store_true", help="Enable template-guided free-walk mode (alias of --free-walk)")
    # Plan-1 open-space sampling (epsilon-ignore mask)
    p.add_argument("--open-eps", type=float, default=0.1, help="Per-step probability to ignore dataset mask and sample in full block×rxn space")
    p.add_argument("--open-topk-block", type=int, default=8, help="Top-K blocks for open sampling")
    p.add_argument("--open-topk-rxn", type=int, default=4, help="Top-K reactions for open sampling")
    p.add_argument("--open-temp", type=float, default=1.0, help="Sampling temperature for open sampling")
    p.add_argument("--open-max-retries", type=int, default=3, help="Retries within a step when open-sampled pair is infeasible")
    p.add_argument("--feasibility-filter", type=str, default="rdkit", choices=["none", "rdkit", "onnx"], help="Feasibility filter for open sampling")
    p.add_argument("--feasibility-onnx-path", type=str, default="lib/uspto_filter_model.onnx", help="Path to ONNX feasibility model (when --feasibility-filter onnx)")
    p.add_argument("--novelty-db", nargs="*", default=[], help="Reference SMILES files (SMI/CSV) used to compute novelty shaping against")
    p.add_argument("--novelty-weight", type=float, default=0.0, help="Weight for novelty shaping added to terminal reward (novelty=1-maxSim)")
    p.add_argument("--use-backward-policy", action="store_true", help="Estimate backward log-prob via inbound degree (dataset graph)")
    p.add_argument("--sub-tb-k", type=int, default=0, help="If >0, add TB residual using only last K steps (sub-trajectory balance)")
    # Learned backward policy options
    p.add_argument("--pb-learned", action="store_true", help="Use learned child-conditioned P_B in TB loss instead of inbound-degree proxy")
    p.add_argument("--pb-source-aware", action="store_true", help="Condition P_B on action source id (0:internal,1:template,2:free-connect)")
    p.add_argument("--pb-logsumexp", action="store_true", help="Use log-sum-exp over candidate parents to compute log P_B(child)")
    p.add_argument("--pb-candidate-cap", type=int, default=64, help="Cap number of candidate parents per child for PB marginalization")
    p.add_argument("--pb-open-topk-block", type=int, default=16, help="Top-K blocks for PB marginalization when child has no inbound edges")
    p.add_argument("--pb-open-topk-rxn", type=int, default=8, help="Top-K reactions per block for PB marginalization when no inbound edges")
    p.add_argument("--pb-bc-weight", type=float, default=0.0, help="Weight of PB BC auxiliary loss per episode (uses child-only marginalization)")
    p.add_argument("--pb-buffer-jsonl", type=str, default="", help="If set, append child states from open transitions to this JSONL for PB supervision")
    # External blocks for template expansion (second reactant)
    p.add_argument("--extra-blocks-csv", type=str, default="data/building_blocks_frag_mw250.csv", help="CSV with column 'smiles' of external blocks")
    p.add_argument("--extra-blocks-cap", type=int, default=20000, help="Max number of external blocks to load")
    p.add_argument("--extra-blocks-prob", type=float, default=0.7, help="Probability to attach an external block when using template expansion")
    # No-immediate-undo guard (default on)
    p.add_argument("--no-immediate-undo", dest="no_immediate_undo", action="store_true", help="Disallow steps that immediately reduce heavy atoms vs current state")
    p.add_argument("--allow-immediate-undo", dest="no_immediate_undo", action="store_false", help="Allow immediate undo (disable guard)")
    p.set_defaults(no_immediate_undo=True, perstep_plantain=True, use_local_reward=True)
    # Metrics/Success accounting
    p.add_argument("--count-open-as-success", action="store_true", help="Count template/free-walk transitions as successes in success_rate")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    # DDP init (optional)
    if args.distributed == "ddp" and device.type == "cuda":
        import torch.distributed as dist
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_init_method)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # Build dataset graph (actions feasible set)
    fwd_df = build_forward_dataset(
        args.input,
        args.forward,
        skip_start_steps=True,
        rebuild=args.rebuild_forward,
        max_block_mw=args.max_block_mw,
    )
    dataset = ForwardTrajectoryDataset(fwd_df)
    known_states_set = set([s for s in fwd_df["state_smiles"].astype(str).tolist() if isinstance(s, str) and s])

    # Infer model dims from checkpoint if present
    inferred_hidden = 256
    inferred_layers = 3
    ckpt_obj = None
    if os.path.exists(args.checkpoint):
        try:
            ckpt_obj = torch.load(args.checkpoint, map_location=device)
            if isinstance(ckpt_obj, dict):
                if "hidden_dim" in ckpt_obj:
                    inferred_hidden = int(ckpt_obj["hidden_dim"])
                if "num_gnn_layers" in ckpt_obj:
                    inferred_layers = int(ckpt_obj["num_gnn_layers"])
                # Fallback: try to infer hidden dim from a known weight tensor shape
                if "model_state" in ckpt_obj and isinstance(ckpt_obj["model_state"], dict):
                    sd = ckpt_obj["model_state"]
                    w = sd.get("state_encoder.convs.0.lin.weight")
                    if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
                        inferred_hidden = int(w.shape[0])
                    # Try to infer number of layers
                    layer_count = sum(1 for k in sd.keys() if k.startswith("state_encoder.convs.") and k.endswith(".lin.weight"))
                    if layer_count > 0:
                        inferred_layers = int(layer_count)
        except Exception:
            ckpt_obj = None

    hidden_dim = int(args.hidden_dim) if int(args.hidden_dim) > 0 else inferred_hidden
    num_layers = int(args.num_gnn_layers) if int(args.num_gnn_layers) > 0 else inferred_layers

    # Model + conditional policy (match dims with checkpoint when possible)
    base = SynthPolicyNet(
        node_feature_dim=dataset.node_feature_dim,
        hidden_dim=hidden_dim,
        num_building_blocks=len(dataset.block_vocab.itos),
        num_reaction_templates=len(dataset.rxn_vocab.itos),
        num_gnn_layers=num_layers,
        dropout=float(args.dropout),
        share_encoders=bool(args.share_encoders),
        use_l2_normalization=bool(args.use_l2_norm),
        initial_temperature=float(args.temperature),
    ).to(device)
    if ckpt_obj is not None:
        # Shape-safe loading: skip params whose shapes don't match current model (e.g., vocab size changes)
        try:
            src_sd = ckpt_obj.get("model_state", ckpt_obj)
            dst_sd = base.state_dict()
            filtered = {}
            skipped = []
            for k, v in src_sd.items():
                if k in dst_sd:
                    try:
                        if hasattr(v, "shape") and hasattr(dst_sd[k], "shape") and (tuple(v.shape) == tuple(dst_sd[k].shape)):
                            filtered[k] = v
                        else:
                            skipped.append(k)
                    except Exception:
                        skipped.append(k)
            if skipped:
                print({"checkpoint_partial_load_skipped": skipped[:10], "skipped_count": len(skipped)})
            base.load_state_dict(filtered, strict=False)
        except Exception:
            # Fallback to best-effort loading
            base.load_state_dict(ckpt_obj.get("model_state", ckpt_obj), strict=False)

    # Protein encoder and pool of protein sequences from pActivity CSV or FASTA
    import pandas as pd
    # Use local ESM2 by default for protein encoding
    from LeadGFlowNet.protein_encoder import Esm2ProteinEncoder
    prot_enc = Esm2ProteinEncoder(model_name="lib/models--facebook--esm2_t30_150M_UR50D").to(device)
    protein_dim = int(getattr(prot_enc, "out_dim", 256))
    proteins: List[str] = []
    
    # Priority 1: Direct protein sequence string
    if getattr(args, "protein_sequence", ""):
        seq = str(args.protein_sequence).strip()
        if seq:
            proteins = [seq]
    
    # Priority 2: FASTA file or FASTA string
    if not proteins and getattr(args, "protein_fasta", ""):
        fasta_input = str(args.protein_fasta).strip()
        if os.path.exists(fasta_input):
            # Read from file
            try:
                with open(fasta_input, 'r', encoding='utf-8') as f:
                    current_seq = []
                    for line in f:
                        line = line.strip()
                        if line.startswith('>'):
                            if current_seq:
                                seq = ''.join(current_seq)
                                if seq:
                                    proteins.append(seq)
                            current_seq = []
                        elif line:
                            current_seq.append(line)
                    if current_seq:
                        seq = ''.join(current_seq)
                        if seq:
                            proteins.append(seq)
            except Exception as e:
                print({"fasta_read_error": str(e)})
        else:
            # Treat as direct FASTA sequence string (without header)
            # Remove common FASTA header markers if present
            seq = fasta_input.replace('>', '').strip()
            # Take first line if multi-line, or use as-is
            if '\n' in seq:
                seq = seq.split('\n')[0].strip()
            if seq and all(c in 'ACDEFGHIKLMNPQRSTVWYX*' for c in seq.upper()):
                proteins = [seq]
    
    # Priority 3: pActivity CSV
    if not proteins and os.path.exists(args.pactivity):
        try:
            pact_df = pd.read_csv(args.pactivity)
            if {"protein_sequence"}.issubset(set(pact_df.columns)):
                proteins = (
                    pact_df["protein_sequence"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().drop_duplicates().tolist()
                )
        except Exception:
            proteins = []
    
    # Fallback to a simple dummy if none found (rare)
    if not proteins:
        proteins = ["MQDRVKRPMNAFIVWSRDQRRKMALEN"]
    _protein_cache: Dict[str, torch.Tensor] = {}
    def get_protein_emb(seq: str) -> torch.Tensor:
        if seq in _protein_cache:
            return _protein_cache[seq]
        emb = prot_enc.encode_sequence(seq)
        _protein_cache[seq] = emb
        return emb

    # Build conditional policy AFTER knowing protein embedding dim
    cond_policy = ConditionalSynthPolicy(base, protein_dim=protein_dim).to(device)
    setattr(cond_policy, "use_rxn_first", bool(args.rxn_first))
    if args.distributed == "ddp" and device.type == "cuda":
        from torch.nn.parallel import DistributedDataParallel as DDP
        cond_policy = DDP(cond_policy, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
    log_z = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))

    # Auto-detect Plantain pocket when requested and not provided
    plantain_pocket = str(getattr(args, "plantain_pocket", ""))
    use_plantain = bool(getattr(args, "use_plantain", False))
    if use_plantain and not plantain_pocket:
        try:
            import glob
            import os as _os
            # Expect exactly one subdir test/<PDBID>/ and inside a *_pocket.pdb
            tests = sorted([d for d in glob.glob("test/*") if _os.path.isdir(d)])
            for d in tests:
                cand = sorted(glob.glob(_os.path.join(d, "*_pocket.pdb")))
                if cand:
                    plantain_pocket = cand[0]
                    break
        except Exception:
            plantain_pocket = ""

    # Reward controller (QSAR or PLANTAIN)
    # Enable QSAR reward if QSAR checkpoint is provided and not using docking guidance
    use_qsar_reward_flag = getattr(args, "use_qsar_reward", False)
    if not use_qsar_reward_flag and args.qsar_checkpoint and os.path.exists(args.qsar_checkpoint):
        # Auto-enable QSAR reward if checkpoint exists and no explicit flag set
        use_qsar_reward_flag = True
    
    rc = MixedRewardController(
        qsar_checkpoint=(args.qsar_checkpoint if (use_qsar_reward_flag or not getattr(args, "use_docking_guidance", False)) else None),
        device=device,
        alpha_start=1.0,
        alpha_end=1.0,
        total_steps=1,
        add_qed=float(getattr(args, "add_qed", 0.3)),
        sub_sa=float(getattr(args, "sub_sa", 0.05)),
        dock_temp=1.0,
        lipinski_penalty=float(getattr(args, "lipinski_penalty", 0.1)),
        use_plantain=(getattr(args, "use_docking_guidance", False) and str(getattr(args, "docking_model", "plantain")).lower()=="plantain"),
        plantain_pocket_pdb=(plantain_pocket or None),
        plantain_device=str(getattr(args, "plantain_device", "auto")),
        plantain_scale=float(getattr(args, "plantain_scale", 10.0)),
        # Vina refine (enable when requested)
        use_vina=bool(getattr(args, "use_vina", False)),
        vina_pocket_pdb=(plantain_pocket or None),
        vina_center=None,
        vina_box_size=float(getattr(args, "vina_box_size", 22.0)),
        vina_exhaustiveness=int(getattr(args, "vina_exhaustiveness", 32)),
        vina_top_k=int(getattr(args, "vina_top_k", 1)),
        vina_full_dock_th=float(getattr(args, "vina_full_dock_th", -3.0)),
        vina_obabel_bin=str(getattr(args, "vina_obabel_bin", "/usr/local/bin/obabel")),
        vina_strict=bool(getattr(args, "vina_strict", False)),
        vina_pdbqt_dir=str(getattr(args, "vina_pdbqt_dir", "runs/vina_pdbqt")),
        vina_reward_smooth=float(getattr(args, "vina_reward_smooth", 0.0)),
        vina_weight=float(getattr(args, "vina_weight", 1.0)),
    )

    # Reference scaffolds storage and loader (reload each epoch to pick up new auto-ref entries)
    ref_scaff_fps: list = []
    def _reload_ref_scaff_fps() -> None:
        nonlocal ref_scaff_fps
        ref_scaff_fps = []
        try:
            ref_path = str(getattr(args, "ref_ligands", ""))
            if ref_path and os.path.exists(ref_path):
                fps = []
                with open(ref_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        s = line.strip().split("\t")[0].split(",")[0].strip().strip('"').strip("'")
                        if not s or s.lower()=="smiles":
                            continue
                        m = Chem.MolFromSmiles(s)
                        if m is None:
                            continue
                        scaf = MurckoScaffold.GetScaffoldForMol(m)
                        if scaf is None:
                            continue
                        fp = AllChem.GetMorganFingerprintAsBitVect(scaf, radius=2, nBits=2048)
                        fps.append(fp)
                ref_scaff_fps = fps
        except Exception:
            ref_scaff_fps = []

    def compute_scaffold_reward(smiles: str) -> float:
        if not ref_scaff_fps:
            return 0.0
        try:
            m = Chem.MolFromSmiles(smiles)
            if m is None:
                return 0.0
            scaf = MurckoScaffold.GetScaffoldForMol(m)
            if scaf is None:
                return 0.0
            fp = AllChem.GetMorganFingerprintAsBitVect(scaf, radius=2, nBits=2048)
            best = 0.0
            for rf in ref_scaff_fps[:2000]:
                try:
                    sim = float(DataStructs.TanimotoSimilarity(fp, rf))
                except Exception:
                    sim = 0.0
                if sim > best:
                    best = sim
            return best
        except Exception:
            return 0.0

    def local_structure_score(smiles: str) -> float:
        # Simple sanity-based local shaping: penalize invalid or fragmented structures
        try:
            m = Chem.MolFromSmiles(smiles)
            if m is None:
                return -1.0
            if m.GetNumBonds() == 0:
                return -0.5
            frags = Chem.GetMolFrags(m)
            if len(frags) > 1:
                return -0.2
            return 0.0
        except Exception:
            return -0.5

    # Optimizer
    # Use higher LR for logZ to accelerate normalization
    optim = Adam([
        {"params": cond_policy.parameters(), "lr": float(args.lr)},
        {"params": [log_z], "lr": float(args.lr) * 10.0},
    ])

    # Normalize flags: template-walk is an alias to free-walk
    if getattr(args, "template_walk", False):
        setattr(args, "free_walk", True)

    # Auto-detect default template file if not provided (prefer top100 CSV then Excel)
    if not getattr(args, "template_csv", ""):
        candidates = [
            "data/top100/template_top100.csv",
            "data/top100/template_top100.xlsx",
            "data/Top100/template_top100.csv",
            "data/Top100/template_top100.xlsx",
            "data/top100/常见反应模板top100.csv",
            "data/top100/原始的USPTO-31k反应模版.csv",
        ]
        for _pth in candidates:
            if os.path.exists(_pth):
                args.template_csv = _pth
                print({"template_csv_auto": args.template_csv})
                break

    # Optional template library (only used when template-walk/free-walk is on)
    template_lib = None
    template_prob = max(0.0, min(1.0, float(getattr(args, "template_prob", 0.0))))
    use_template_walk = bool(getattr(args, "free_walk", False))
    if use_template_walk and template_prob > 0.0 and isinstance(getattr(args, "template_csv", None), str) and args.template_csv:
        if os.path.exists(args.template_csv):
            try:
                template_lib = TemplateLibrary.from_csv(args.template_csv, max_rows=int(getattr(args, "template_max_rows", 5000)))
                print({"template_csv": args.template_csv, "templates_loaded": len(getattr(template_lib, "compiled", [])), "template_prob": template_prob})
            except Exception as e:
                print({"template_load_error": str(e)})

    # Load external blocks from CSV (column 'smiles')
    extra_blocks: List[str] = []
    try:
        eb_path = str(getattr(args, "extra_blocks_csv", ""))
        if eb_path and os.path.exists(eb_path):
            import pandas as pd
            df_blk = pd.read_csv(eb_path, usecols=["smiles"])  # expect 'smiles' column
            col = df_blk["smiles"].dropna().astype(str).str.strip()
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors
                seen = set(); kept: List[str] = []
                for s in col.tolist():
                    if not s or s in seen:
                        continue
                    m = Chem.MolFromSmiles(s)
                    if m is None:
                        continue
                    mw = Descriptors.MolWt(m)
                    if mw <= 200.0:
                        can = Chem.MolToSmiles(m)
                        if can not in seen:
                            seen.add(can)
                            kept.append(can)
                extra_blocks = kept
            except Exception:
                # Fallback: unique non-empty
                seen = set(); tmp = []
                for s in col.tolist():
                    if s and s not in seen:
                        seen.add(s); tmp.append(s)
                extra_blocks = tmp
            capn = max(1, int(getattr(args, "extra_blocks_cap", 20000)))
            if len(extra_blocks) > capn:
                extra_blocks = extra_blocks[:capn]
            print({"extra_blocks_loaded": len(extra_blocks), "src": eb_path})
    except Exception as e:
        print({"extra_blocks_error": str(e)})

    # Simple online loop sampling from dataset space
    from torch_geometric.data import Data
    import numpy as np

    def sample_start_states(df) -> List[str]:
        produced = set(df["result_smiles"].astype(str).tolist())
        candidates = [s for s in df["state_smiles"].astype(str).tolist() if s and s not in produced]
        if not candidates:
            candidates = [""]
        random.shuffle(candidates)
        cap = max(1, int(getattr(args, "start_states_cap", 256)))
        return candidates[:cap]

    start_states = sample_start_states(fwd_df)
    # Build a degree-weighted pool of start states to reduce early fails
    _deg = fwd_df.groupby("state_smiles", sort=False).size().to_dict()
    start_pool: List[str] = []
    min_deg_pref = max(0, int(getattr(args, "min_start_degree", 2)))
    for st in start_states:
        d = int(_deg.get(st, 0))
        if d < min_deg_pref:
            repeat = 1
        else:
            repeat = max(2, min(12, d))
        start_pool.extend([st] * repeat)

    def pick_start_state_with_edges(max_tries: int = 5) -> str:
        for _ in range(max(1, int(max_tries))):
            st = random.choice(start_pool) if start_pool else random.choice(start_states)
            if not st:
                continue
            es = fwd_df[fwd_df["state_smiles"] == st]
            if not es.empty:
                return st
        return random.choice(start_states)

    # Resolve project root for absolute output paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Prepare auto-reference collection configuration
    runs_dir_default = os.path.join(project_root, "runs")
    try:
        os.makedirs(runs_dir_default, exist_ok=True)
    except Exception:
        pass
    auto_ref_th = float(getattr(args, "auto_ref_vina_th", 0.0))
    auto_ref_out = str(getattr(args, "auto_ref_out", "")).strip()
    if not auto_ref_out:
        auto_ref_out = os.path.join(runs_dir_default, "ref_auto_top.smi")
    elif not os.path.isabs(auto_ref_out):
        auto_ref_out = os.path.join(project_root, auto_ref_out)
    # If requested, enable scaffold reward using auto-ref file (finetune stage)
    if auto_ref_th < 0.0 and bool(getattr(args, "auto_ref_use_scaffold", False)):
        try:
            # Only set defaults if user hasn't explicitly provided
            if not bool(getattr(args, "use_scaffold_reward", False)):
                setattr(args, "use_scaffold_reward", True)
            if not str(getattr(args, "ref_ligands", "")).strip():
                setattr(args, "ref_ligands", auto_ref_out)
            # Switch to finetune stage to activate scaffold reward path
            setattr(args, "training_stage", "finetune")
            # Apply scaffold weight override if provided
            if float(getattr(args, "auto_ref_scaffold_weight", 0.2)) != float(getattr(args, "scaffold_weight", 0.2)):
                setattr(args, "scaffold_weight", float(getattr(args, "auto_ref_scaffold_weight", 0.2)))
            print({
                "auto_ref_scaffold_enabled": {
                    "ref_path": str(getattr(args, "ref_ligands", auto_ref_out)),
                    "weight": float(getattr(args, "scaffold_weight", 0.2))
                }
            })
        except Exception:
            pass
    # Preload existing references to dedupe across runs
    auto_ref_seen = set()
    try:
        if os.path.exists(auto_ref_out):
            with open(auto_ref_out, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip().split("\t")[0].split(",")[0].strip().strip('"').strip("'")
                    if not s:
                        continue
                    m = Chem.MolFromSmiles(s)
                    if m is None:
                        continue
                    can = Chem.MolToSmiles(m)
                    if can:
                        auto_ref_seen.add(can)
    except Exception:
        auto_ref_seen = set()

    for epoch in range(1, args.epochs + 1):
        cond_policy.train()
        # Reload reference scaffold fingerprints at each epoch if scaffold reward is active
        if bool(getattr(args, "use_scaffold_reward", False)) and str(getattr(args, "training_stage", "pretrain")) == "finetune":
            _reload_ref_scaff_fps()
            try:
                print({"scaffold_refs": {"count": len(ref_scaff_fps), "path": str(getattr(args, "ref_ligands", ""))}})
            except Exception:
                pass
        # Linear schedules
        t = (epoch - 1) / max(1, (args.epochs - 1))
        tf_prob = float(args.tf_start) + (float(args.tf_end) - float(args.tf_start)) * t
        samp_temp = float(args.temp_start) + (float(args.temp_end) - float(args.temp_start)) * t
        tb_topk_b = int(round(float(args.topk_block_start) + (float(args.topk_block_end) - float(args.topk_block_start)) * t))
        tb_topk_r = int(round(float(args.topk_rxn_start) + (float(args.topk_rxn_end) - float(args.topk_rxn_start)) * t))
        total_loss = 0.0
        valid_loss_count = 0
        epoch_rewards: List[float] = []
        # Per-episode Vina energies (min and raw) for plotting vs episodes
        epoch_vina_min: List[float] = []
        epoch_vina_raw: List[float] = []
        epoch_terminals: set[str] = set()
        epoch_counts: Counter[str] = Counter()
        success_acc: float = 0.0
        success_n: int = 0
        # Open-space metrics accumulation
        epoch_template_steps: int = 0
        epoch_freewalk_steps: int = 0
        epoch_dataset_success_steps: int = 0
        # Roll out episodes and apply TB loss
        for _ in range(max(1, args.episodes_per_epoch)):
            # Pick a start and roll out up to max steps within dataset edges
            state = pick_start_state_with_edges(max_tries=5)
            protein = random.choice(proteins)
            protein_emb = get_protein_emb(protein)

            # Fallback graph feature dimension
            from SynthPolicyNet.data_utils import build_graph_from_smiles
            node_dim = int(dataset.node_feature_dim)

            # Encode block graphs with current model weights (refresh each episode)
            base_mod = getattr(getattr(cond_policy, "module", cond_policy), "base")
            valid_block_graphs = []
            for g0 in dataset.block_graphs:
                if g0 is None:
                    from torch_geometric.data import Data as TGData
                    valid_block_graphs.append(
                        TGData(
                            x=torch.zeros((1, node_dim), dtype=torch.float32),
                            edge_index=torch.zeros((2, 0), dtype=torch.long),
                        )
                    )
                else:
                    valid_block_graphs.append(g0)
            block_embs = base_mod.encode_blocks(valid_block_graphs, device=device)

            log_pf_total = torch.tensor(0.0, device=device)
            log_pb_total = torch.tensor(0.0, device=device)
            successful_steps = 0
            attempted_steps = 0
            # Per-episode open-space counters
            template_steps = 0
            freewalk_steps = 0
            dataset_success_steps = 0
            # Accumulate local reward shaping
            local_accum = 0.0

            for _step in range(max(1, args.max_steps)):
                # Encode current state
                g = build_graph_from_smiles(state)
                if g is None:
                    from torch_geometric.data import Data as TGData
                    g = TGData(
                        x=torch.zeros((1, node_dim), dtype=torch.float32),
                        edge_index=torch.zeros((2, 0), dtype=torch.long),
                    )
                if not hasattr(g, "batch"):
                    g.batch = torch.zeros((g.x.size(0),), dtype=torch.long)
                g = g.to(device)

                h_state_block = cond_policy.compute_h_state_block(g, protein_emb)

                # Build feasibility filters from dataset graph at this state unless open-eps kicks in
                use_open = False
                if not bool(getattr(args, "free_walk", False)):
                    try:
                        if random.random() < float(getattr(args, "open_eps", 0.1)):
                            use_open = True
                    except Exception:
                        use_open = False
                edges_state = fwd_df[fwd_df["state_smiles"] == state] if (not bool(getattr(args, "free_walk", False)) and not use_open) else fwd_df.iloc[0:0]
                # Guard: no-immediate-undo (filter edges that reduce heavy atoms vs current state)
                if not edges_state.empty and bool(getattr(args, "no_immediate_undo", True)):
                    try:
                        m_curr = Chem.MolFromSmiles(state) if state else None
                        heavy_curr = int(m_curr.GetNumHeavyAtoms()) if m_curr is not None else None
                    except Exception:
                        heavy_curr = None

                    if heavy_curr is not None:
                        def _ok_row(row) -> bool:
                            try:
                                rs = str(row.get("result_smiles", ""))
                                m = Chem.MolFromSmiles(rs)
                                if m is None:
                                    return True
                                return int(m.GetNumHeavyAtoms()) >= int(heavy_curr)
                            except Exception:
                                return True
                        edges_state = edges_state[edges_state.apply(_ok_row, axis=1)]

                # Pure free-connect branch (no templates): only in template-walk mode
                use_free_connect = False
                if bool(getattr(args, "free_walk", False)) and bool(getattr(args, "free_connect", False)):
                    use_free_connect = True if edges_state.empty else (random.random() < float(getattr(args, "free_connect_prob", 1.0)))
                if use_free_connect:
                    cand_products_fc: List[str] = []
                    try:
                        # Sample external blocks
                        if 'extra_blocks' in locals() and extra_blocks:
                            k = min(int(getattr(args, "free_connect_sample_blocks", 1024)), len(extra_blocks))
                            import random as _rfc
                            blocks_sampled = _rfc.sample(extra_blocks, k=k) if len(extra_blocks) > k else list(extra_blocks)
                            pair_tries = max(1, int(getattr(args, "free_connect_tries", 64)))
                            # Try to connect with a subset until first success
                            for bs in blocks_sampled:
                                prod = _connect_mols_random(state, bs, max_pair_tries=pair_tries)
                                if prod:
                                    # canonicalize
                                    m_fw = Chem.MolFromSmiles(prod)
                                    if m_fw is not None:
                                        prod = Chem.MolToSmiles(m_fw)
                                    cand_products_fc.append(prod)
                                    break
                    except Exception:
                        cand_products_fc = []
                    if cand_products_fc:
                        try:
                            import random as _rpick
                            pick = _rpick.choice(cand_products_fc)
                            state = pick
                            # Count as free-walk open transition
                            freewalk_steps += 1
                            if bool(getattr(args, "count_open_as_success", False)):
                                attempted_steps += 1
                                successful_steps += 1
                            # PB buffer logging for supervision (source=2)
                            if getattr(args, "pb_buffer_jsonl", ""):
                                try:
                                    os.makedirs(os.path.dirname(args.pb_buffer_jsonl) or ".", exist_ok=True)
                                    with open(args.pb_buffer_jsonl, "a", encoding="utf-8") as f:
                                        f.write(json.dumps({"child": state, "source": 2}) + "\n")
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        continue

                # Template expansion branch: with probability or when no dataset edges (only in template-walk mode)
                use_template = False
                if bool(getattr(args, "free_walk", False)) and template_lib is not None:
                    if edges_state.empty:
                        use_template = True
                    else:
                        import random as _rnd
                        if _rnd.random() < template_prob:
                            use_template = True
                if use_template:
                    try:
                        blk = None
                        prob_ext = max(0.0, min(1.0, float(getattr(args, "extra_blocks_prob", 0.7))))
                        cand_products: List[str] = []
                        if 'extra_blocks' in locals() and extra_blocks and prob_ext > 0.0:
                            import random as _rb
                            if _rb.random() < prob_ext:
                                # Try matching templates to a pooled subset of blocks (improves hit rate)
                                cand_products = template_lib.propose_products_with_pool(
                                    state_smiles=state,
                                    block_pool=extra_blocks,
                                    try_limit_templates=max(1, int(getattr(args, "template_try_templates", 128))),
                                    sample_blocks=min(int(getattr(args, "template_sample_blocks", 2048)), len(extra_blocks)),
                                    max_products_per_template=1,
                                )
                        if not cand_products:
                            # Fallback: single-block or no-block attempt
                            if 'extra_blocks' in locals() and extra_blocks and prob_ext > 0.0:
                                blk = _rb.choice(extra_blocks)
                            cand_products = template_lib.propose_products(
                                state_smiles=state,
                                block_smiles=blk,
                                try_limit=max(1, int(getattr(args, "template_try_templates", 128))),
                                max_products_per_template=1,
                            )
                    except Exception:
                        cand_products = []
                    snapped: list[str] = []
                    for prod in cand_products:
                        if prod in known_states_set:
                            if not fwd_df[fwd_df["state_smiles"] == prod].empty:
                                snapped.append(prod)
                    if snapped:
                        import random as _r2
                        new_state = _r2.choice(snapped)
                        attempted_steps += 1
                        successful_steps += 1
                        dataset_success_steps += 1
                        template_steps += 1
                        state = new_state
                        continue
                    # Free-walk acceptance: if enabled and we have any candidate products, move to one product directly
                    if bool(getattr(args, "free_walk", False)) and cand_products:
                        try:
                            # Canonicalize a random product
                            import random as _rfw
                            pick = _rfw.choice(cand_products)
                            m_fw = Chem.MolFromSmiles(pick)
                            if m_fw is not None:
                                pick = Chem.MolToSmiles(m_fw)
                            state = pick
                            template_steps += 1
                            freewalk_steps += 1
                            if bool(getattr(args, "count_open_as_success", False)):
                                attempted_steps += 1
                                successful_steps += 1
                            # PB buffer logging for supervision (source=1)
                            if getattr(args, "pb_buffer_jsonl", ""):
                                try:
                                    os.makedirs(os.path.dirname(args.pb_buffer_jsonl) or ".", exist_ok=True)
                                    with open(args.pb_buffer_jsonl, "a", encoding="utf-8") as f:
                                        f.write(json.dumps({"child": state, "source": 1}) + "\n")
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        continue
                if edges_state.empty and not bool(getattr(args, "free_walk", False)) and not use_open:
                    break

                # Plan-1: Open-space sampling when use_open is True
                if use_open and not bool(getattr(args, "free_walk", False)):
                    base_mod = getattr(getattr(cond_policy, "module", cond_policy), "base")
                    # Unconditional reaction logits and block logits
                    uncond_rxn_logits = base_mod.uncond_rxn_head(h_state_block).squeeze(0)
                    rxn_probs = torch.softmax(uncond_rxn_logits / max(1e-6, float(getattr(args, "open_temp", 1.0))), dim=-1)
                    rxn_probs = torch.nan_to_num(rxn_probs, nan=0.0, posinf=0.0, neginf=0.0)
                    rxn_probs = torch.clamp(rxn_probs, min=0.0)
                    Kr_open = int(max(1, min(int(getattr(args, "open_topk_rxn", 4)), rxn_probs.numel())))
                    vals_r, inds_r = torch.topk(rxn_probs, k=Kr_open)
                    vals_r = torch.nan_to_num(vals_r, nan=0.0, posinf=0.0, neginf=0.0)
                    vals_r = torch.clamp(vals_r, min=0.0)
                    tot_r = float(torch.sum(vals_r).item())
                    if tot_r > 0:
                        vals_r = vals_r / tot_r
                    # Block logits unconditional
                    block_logits = base_mod.compute_block_logits(h_state_block, block_embs).squeeze(0)
                    block_probs = torch.softmax(block_logits / max(1e-6, float(getattr(args, "open_temp", 1.0))), dim=-1)
                    block_probs = torch.nan_to_num(block_probs, nan=0.0, posinf=0.0, neginf=0.0)
                    block_probs = torch.clamp(block_probs, min=0.0)
                    Kb_open = int(max(1, min(int(getattr(args, "open_topk_block", 8)), block_probs.numel())))
                    vals_b, inds_b = torch.topk(block_probs, k=Kb_open)
                    vals_b = torch.nan_to_num(vals_b, nan=0.0, posinf=0.0, neginf=0.0)
                    vals_b = torch.clamp(vals_b, min=0.0)
                    tot_b = float(torch.sum(vals_b).item())
                    if tot_b > 0:
                        vals_b = vals_b / tot_b
                    max_retries = int(max(0, getattr(args, "open_max_retries", 3)))
                    tried = 0
                    picked = False
                    while tried <= max_retries:
                        tried += 1
                        # sample rxn and block from open top-k
                        ir = int(inds_r[int(torch.multinomial(vals_r if tot_r > 0 else torch.ones_like(vals_r) / max(1, vals_r.numel()), 1))].item())
                        ib = int(inds_b[int(torch.multinomial(vals_b if tot_b > 0 else torch.ones_like(vals_b) / max(1, vals_b.numel()), 1))].item())
                        # feasibility check
                        feas = True
                        filt = str(getattr(args, "feasibility_filter", "rdkit")).lower()
                        if filt != "none":
                            try:
                                # Build a one-step by applying dataset reaction template string for r_idx on state+block
                                # We don't have explicit rxn SMARTS here; only template index string from vocab
                                # Use dataset transition feasibility as proxy: if (state, ib, ir) exists somewhere in any state, accept; else try rdkit synthesis via dataset edges is not possible here.
                                feas = True
                            except Exception:
                                feas = False
                        if feas:
                            try:
                                block_smi_sel = str(dataset.block_vocab.itos[ib])
                                rxn_tpl_sel = str(dataset.rxn_vocab.itos[ir])
                                row_any = fwd_df[(fwd_df["action_building_block"] == block_smi_sel) & (fwd_df["action_reaction_template"] == rxn_tpl_sel)].head(1)
                                if not row_any.empty:
                                    state = str(row_any.iloc[0]["result_smiles"]) or state
                                    attempted_steps += 1
                                    successful_steps += 1
                                    dataset_success_steps += 1
                                    picked = True
                                    break
                            except Exception:
                                pass
                    if picked:
                        continue
                block_stoi = getattr(dataset.block_vocab, "stoi", {})
                rxn_stoi = getattr(dataset.rxn_vocab, "stoi", {})
                allowed_pairs = []
                allowed_blocks_set = set()
                allowed_rxns_set = set()
                try:
                    for _, row_s in edges_state.iterrows():
                        b_smi = str(row_s.get("action_building_block", ""))
                        r_tpl = str(row_s.get("action_reaction_template", ""))
                        b_idx_allowed = block_stoi.get(b_smi) if isinstance(block_stoi, dict) else None
                        r_idx_allowed = rxn_stoi.get(r_tpl) if isinstance(rxn_stoi, dict) else None
                        if b_idx_allowed is None or r_idx_allowed is None:
                            continue
                        allowed_pairs.append((int(b_idx_allowed), int(r_idx_allowed)))
                        allowed_blocks_set.add(int(b_idx_allowed))
                        allowed_rxns_set.add(int(r_idx_allowed))
                except Exception:
                    pass

                if getattr(cond_policy, "use_rxn_first", False):
                    # Reaction-first: stochastic/deterministic selection
                    uncond_rxn_logits = base_mod.uncond_rxn_head(h_state_block).squeeze(0)
                    rxn_probs = torch.softmax(uncond_rxn_logits / max(1e-6, float(samp_temp)), dim=-1)
                    rxn_probs = torch.nan_to_num(rxn_probs, nan=0.0, posinf=0.0, neginf=0.0)
                    rxn_probs = torch.clamp(rxn_probs, min=0.0)
                    # Filter to allowed rxns at this state
                    if allowed_rxns_set:
                        mask = torch.zeros_like(rxn_probs)
                        for rid in allowed_rxns_set:
                            if 0 <= rid < mask.numel():
                                mask[rid] = 1.0
                        rxn_probs = rxn_probs * mask
                    rxn_probs = torch.nan_to_num(rxn_probs, nan=0.0, posinf=0.0, neginf=0.0)
                    rxn_probs = torch.clamp(rxn_probs, min=0.0)
                    s = float(torch.sum(rxn_probs).item())
                    if s > 0:
                        rxn_probs = rxn_probs / s
                    # Fallback when nothing feasible
                    if s <= 0 and not allowed_rxns_set:
                        break
                    # Optional teacher forcing: with probability p, pick rxn uniformly from feasible set
                    tf_p = max(0.0, min(1.0, float(tf_prob)))
                    use_tf_r = (len(allowed_rxns_set) > 0) and (random.random() < tf_p)
                    if use_tf_r:
                        r_idx = int(random.choice(list(allowed_rxns_set)))
                        # For TB, still accumulate model log-prob at chosen index if available
                        if s > 0 and r_idx < rxn_probs.numel():
                            log_pf_total = log_pf_total + torch.log(rxn_probs[r_idx] + 1e-12)
                        else:
                            log_pf_total = log_pf_total + torch.log(torch.tensor(1.0 / max(1, len(allowed_rxns_set)), device=device))
                    elif bool(getattr(args, "train_deterministic", False)):
                        if s <= 0:
                            # Uniform over allowed rxns
                            r_idx = int(random.choice(list(allowed_rxns_set)))
                            log_pf_total = log_pf_total + torch.log(torch.tensor(1.0 / max(1, len(allowed_rxns_set)), device=device))
                        else:
                            r_idx = int(torch.argmax(rxn_probs).item())
                            log_pf_total = log_pf_total + torch.log(rxn_probs[r_idx] + 1e-12)
                    else:
                        Kr = max(1, int(tb_topk_r))
                        if s <= 0:
                            # Sample uniformly from allowed set
                            r_idx = int(random.choice(list(allowed_rxns_set)))
                            log_pf_total = log_pf_total + torch.log(torch.tensor(1.0 / max(1, len(allowed_rxns_set)), device=device))
                        else:
                            # top-K restriction
                            Kr = min(Kr, rxn_probs.numel())
                        vals_r, inds_r = torch.topk(rxn_probs, k=Kr)
                        vals_r = torch.nan_to_num(vals_r, nan=0.0, posinf=0.0, neginf=0.0)
                        vals_r = torch.clamp(vals_r, min=0.0)
                        tot = float(torch.sum(vals_r).item())
                        if tot <= 0:
                            r_idx = int(torch.argmax(rxn_probs).item())
                            log_pf_total = log_pf_total + torch.log(rxn_probs[r_idx] + 1e-12)
                        else:
                            vals_r = vals_r / tot
                            pick = int(torch.multinomial(vals_r, num_samples=1, replacement=False).item())
                            r_idx = int(inds_r[pick].item())
                            log_pf_total = log_pf_total + torch.log((vals_r[pick]) + 1e-12)
                    r_idx_tensor = torch.tensor([r_idx], dtype=torch.long, device=device)
                    block_logits = base_mod.compute_block_logits_given_rxn_h(h_state_block, block_embs, r_idx_tensor).squeeze(0)
                    block_probs = torch.softmax(block_logits / max(1e-6, float(samp_temp)), dim=-1)
                    block_probs = torch.nan_to_num(block_probs, nan=0.0, posinf=0.0, neginf=0.0)
                    block_probs = torch.clamp(block_probs, min=0.0)
                    # Filter to allowed blocks given this rxn
                    sb = float(torch.sum(block_probs).item())
                    if allowed_pairs:
                        allowed_b_for_r = {b for (b, r) in allowed_pairs if r == r_idx}
                        if allowed_b_for_r:
                            maskb = torch.zeros_like(block_probs)
                            for bid in allowed_b_for_r:
                                if 0 <= bid < maskb.numel():
                                    maskb[bid] = 1.0
                            block_probs = block_probs * maskb
                    block_probs = torch.nan_to_num(block_probs, nan=0.0, posinf=0.0, neginf=0.0)
                    block_probs = torch.clamp(block_probs, min=0.0)
                    sb = float(torch.sum(block_probs).item())
                    if sb > 0:
                        block_probs = block_probs / sb
                    # Fallback when nothing feasible
                    if sb <= 0 and not (allowed_pairs and allowed_b_for_r):
                        break
                    # Optional teacher forcing for block given rxn
                    allowed_b_for_r = {b for (b, r) in allowed_pairs if r == r_idx} if allowed_pairs else set()
                    use_tf_b = (len(allowed_b_for_r) > 0) and (random.random() < tf_p)
                    if use_tf_b:
                        b_idx = int(random.choice(list(allowed_b_for_r)))
                        if sb > 0 and b_idx < block_probs.numel():
                            log_pf_total = log_pf_total + torch.log(block_probs[b_idx] + 1e-12)
                        else:
                            log_pf_total = log_pf_total + torch.log(torch.tensor(1.0 / max(1, len(allowed_b_for_r)), device=device))
                    elif bool(getattr(args, "train_deterministic", False)):
                        if sb <= 0 and allowed_pairs and allowed_b_for_r:
                            b_idx = int(random.choice(list(allowed_b_for_r)))
                            log_pf_total = log_pf_total + torch.log(torch.tensor(1.0 / max(1, len(allowed_b_for_r)), device=device))
                        else:
                            b_idx = int(torch.argmax(block_probs).item())
                            log_pf_total = log_pf_total + torch.log(block_probs[b_idx] + 1e-12)
                    else:
                        Kb = max(1, int(tb_topk_b))
                        if sb <= 0 and allowed_pairs and allowed_b_for_r:
                            b_idx = int(random.choice(list(allowed_b_for_r)))
                            log_pf_total = log_pf_total + torch.log(torch.tensor(1.0 / max(1, len(allowed_b_for_r)), device=device))
                        else:
                            Kb = min(Kb, block_probs.numel())
                        vals_b, inds_b = torch.topk(block_probs, k=Kb)
                        vals_b = torch.nan_to_num(vals_b, nan=0.0, posinf=0.0, neginf=0.0)
                        vals_b = torch.clamp(vals_b, min=0.0)
                        totb = float(torch.sum(vals_b).item())
                        if totb <= 0:
                            b_idx = int(torch.argmax(block_probs).item())
                            log_pf_total = log_pf_total + torch.log(block_probs[b_idx] + 1e-12)
                        else:
                            vals_b = vals_b / totb
                            pickb = int(torch.multinomial(vals_b, num_samples=1, replacement=False).item())
                            b_idx = int(inds_b[pickb].item())
                            log_pf_total = log_pf_total + torch.log((vals_b[pickb]) + 1e-12)
                else:
                    # Block-first: stochastic/deterministic selection
                    block_logits = base_mod.compute_block_logits(h_state_block, block_embs).squeeze(0)
                    block_probs = torch.softmax(block_logits / max(1e-6, float(samp_temp)), dim=-1)
                    block_probs = torch.nan_to_num(block_probs, nan=0.0, posinf=0.0, neginf=0.0)
                    block_probs = torch.clamp(block_probs, min=0.0)
                    # Filter to allowed blocks at this state
                    if allowed_blocks_set:
                        maskb = torch.zeros_like(block_probs)
                        for bid in allowed_blocks_set:
                            if 0 <= bid < maskb.numel():
                                maskb[bid] = 1.0
                        block_probs = block_probs * maskb
                    block_probs = torch.nan_to_num(block_probs, nan=0.0, posinf=0.0, neginf=0.0)
                    block_probs = torch.clamp(block_probs, min=0.0)
                    sb = float(torch.sum(block_probs).item())
                    if sb > 0:
                        block_probs = block_probs / sb
                    if sb <= 0 and not allowed_blocks_set:
                        break
                    # Optional teacher forcing for block at state
                    tf_p = max(0.0, min(1.0, float(tf_prob)))
                    use_tf_b0 = (len(allowed_blocks_set) > 0) and (random.random() < tf_p)
                    if use_tf_b0:
                        b_idx = int(random.choice(list(allowed_blocks_set)))
                        if sb > 0 and b_idx < block_probs.numel():
                            log_pf_total = log_pf_total + torch.log(block_probs[b_idx] + 1e-12)
                        else:
                            log_pf_total = log_pf_total + torch.log(torch.tensor(1.0 / max(1, len(allowed_blocks_set)), device=device))
                    elif bool(getattr(args, "train_deterministic", False)):
                        if sb <= 0 and allowed_blocks_set:
                            b_idx = int(random.choice(list(allowed_blocks_set)))
                            log_pf_total = log_pf_total + torch.log(torch.tensor(1.0 / max(1, len(allowed_blocks_set)), device=device))
                        else:
                            b_idx = int(torch.argmax(block_probs).item())
                            log_pf_total = log_pf_total + torch.log(block_probs[b_idx] + 1e-12)
                    else:
                        Kb = max(1, int(tb_topk_b))
                        if sb <= 0 and allowed_blocks_set:
                            b_idx = int(random.choice(list(allowed_blocks_set)))
                            log_pf_total = log_pf_total + torch.log(torch.tensor(1.0 / max(1, len(allowed_blocks_set)), device=device))
                        else:
                            Kb = min(Kb, block_probs.numel())
                        vals_b, inds_b = torch.topk(block_probs, k=Kb)
                        vals_b = torch.nan_to_num(vals_b, nan=0.0, posinf=0.0, neginf=0.0)
                        vals_b = torch.clamp(vals_b, min=0.0)
                        totb = float(torch.sum(vals_b).item())
                        if totb <= 0:
                            b_idx = int(torch.argmax(block_probs).item())
                            log_pf_total = log_pf_total + torch.log(block_probs[b_idx] + 1e-12)
                        else:
                            vals_b = vals_b / totb
                            pickb = int(torch.multinomial(vals_b, num_samples=1, replacement=False).item())
                            b_idx = int(inds_b[pickb].item())
                            log_pf_total = log_pf_total + torch.log((vals_b[pickb]) + 1e-12)
                    b_idx_tensor = torch.tensor([b_idx], dtype=torch.long, device=device)
                    selected_block = block_embs.index_select(0, b_idx_tensor)
                    rxn_logits = base_mod.reaction_head(torch.cat([h_state_block, selected_block], dim=1)).squeeze(0)
                    rxn_probs = torch.softmax(rxn_logits / max(1e-6, float(samp_temp)), dim=-1)
                    rxn_probs = torch.nan_to_num(rxn_probs, nan=0.0, posinf=0.0, neginf=0.0)
                    rxn_probs = torch.clamp(rxn_probs, min=0.0)
                    # Filter to allowed rxns given selected block
                    if allowed_pairs:
                        allowed_r_for_b = {r for (b, r) in allowed_pairs if b == b_idx}
                        if allowed_r_for_b:
                            maskr = torch.zeros_like(rxn_probs)
                            for rid in allowed_r_for_b:
                                if 0 <= rid < maskr.numel():
                                    maskr[rid] = 1.0
                            rxn_probs = rxn_probs * maskr
                    rxn_probs = torch.nan_to_num(rxn_probs, nan=0.0, posinf=0.0, neginf=0.0)
                    rxn_probs = torch.clamp(rxn_probs, min=0.0)
                    sr = float(torch.sum(rxn_probs).item())
                    if sr > 0:
                        rxn_probs = rxn_probs / sr
                    if sr <= 0 and not (allowed_pairs and allowed_r_for_b):
                        break
                    # Optional teacher forcing for rxn given block
                    allowed_r_for_b = {r for (b, r) in allowed_pairs if b == b_idx} if allowed_pairs else set()
                    use_tf_r2 = (len(allowed_r_for_b) > 0) and (random.random() < tf_p)
                    if use_tf_r2:
                        r_idx = int(random.choice(list(allowed_r_for_b)))
                        if sr > 0 and r_idx < rxn_probs.numel():
                            log_pf_total = log_pf_total + torch.log(rxn_probs[r_idx] + 1e-12)
                        else:
                            log_pf_total = log_pf_total + torch.log(torch.tensor(1.0 / max(1, len(allowed_r_for_b)), device=device))
                    elif bool(getattr(args, "train_deterministic", False)):
                        if sr <= 0 and allowed_pairs and allowed_r_for_b:
                            r_idx = int(random.choice(list(allowed_r_for_b)))
                            log_pf_total = log_pf_total + torch.log(torch.tensor(1.0 / max(1, len(allowed_r_for_b)), device=device))
                        else:
                            r_idx = int(torch.argmax(rxn_probs).item())
                            log_pf_total = log_pf_total + torch.log(rxn_probs[r_idx] + 1e-12)
                    else:
                        Kr = max(1, int(tb_topk_r))
                        if sr <= 0 and allowed_pairs and allowed_r_for_b:
                            r_idx = int(random.choice(list(allowed_r_for_b)))
                            log_pf_total = log_pf_total + torch.log(torch.tensor(1.0 / max(1, len(allowed_r_for_b)), device=device))
                        else:
                            Kr = min(Kr, rxn_probs.numel())
                        vals_r, inds_r = torch.topk(rxn_probs, k=Kr)
                        vals_r = torch.nan_to_num(vals_r, nan=0.0, posinf=0.0, neginf=0.0)
                        vals_r = torch.clamp(vals_r, min=0.0)
                        totr = float(torch.sum(vals_r).item())
                        if totr <= 0:
                            r_idx = int(torch.argmax(rxn_probs).item())
                            log_pf_total = log_pf_total + torch.log(rxn_probs[r_idx] + 1e-12)
                        else:
                            vals_r = vals_r / totr
                            pickr = int(torch.multinomial(vals_r, num_samples=1, replacement=False).item())
                            r_idx = int(inds_r[pickr].item())
                            log_pf_total = log_pf_total + torch.log((vals_r[pickr]) + 1e-12)

                # Transition within dataset: find a matching edge (restricted to current state's edges)
                attempted_steps += 1
                try:
                    block_smi_sel = str(dataset.block_vocab.itos[b_idx])
                    rxn_tpl_sel = str(dataset.rxn_vocab.itos[r_idx])
                except Exception:
                    block_smi_sel = ""
                    rxn_tpl_sel = ""
                row_local = edges_state[(edges_state["action_building_block"] == block_smi_sel) &
                                        (edges_state["action_reaction_template"] == rxn_tpl_sel)].head(1)
                if row_local.empty:
                    # Retry a few times by sampling uniformly over feasible pairs at this state
                    retries = max(0, int(getattr(args, "per_step_retries", 2)))
                    matched = False
                    feasible_pairs = [(str(r.get("action_building_block", "")), str(r.get("action_reaction_template", ""))) for _, r in edges_state.iterrows()]
                    for _r in range(retries):
                        if not feasible_pairs:
                            break
                        b_smi_try, r_tpl_try = random.choice(feasible_pairs)
                        row_local = edges_state[(edges_state["action_building_block"] == b_smi_try) &
                                                (edges_state["action_reaction_template"] == r_tpl_try)].head(1)
                        if not row_local.empty:
                            matched = True
                            break
                    if not matched:
                        break
                successful_steps += 1
                dataset_success_steps += 1
                if bool(getattr(args, "use_local_reward", False)):
                    try:
                        local_accum = local_accum + float(local_structure_score(state))
                    except Exception:
                        pass
                # Optional per-step PLANTAIN shaping
                if bool(getattr(args, "perstep_plantain", True)) and bool(getattr(args, "use_plantain", False)) and str(getattr(args, "plantain_pocket", "")):
                    try:
                        # Apply at specified interval
                        if int(max(1, getattr(args, "perstep_plantain_interval", 1))) == 1 or ((_step + 1) % int(max(1, getattr(args, "perstep_plantain_interval", 1))) == 0):
                            from LeadGFlowNet.oracle import _plantain_min_score_for_smiles as _pmin, transform_plantain as _tp
                            ps = _pmin(state, str(getattr(args, "plantain_pocket", "")), device=str(getattr(args, "plantain_device", "auto")))
                            if ps is not None:
                                local_accum = local_accum + float(getattr(args, "perstep_plantain_weight", 0.05)) * float(_tp(ps, scale=float(getattr(args, "plantain_scale", 10.0))))
                    except Exception:
                        pass
                # Compute learned backward log-prob for this step if enabled
                if bool(getattr(args, "pb_learned", False)):
                    try:
                        child_smi = str(row_local.iloc[0]["result_smiles"]) or state
                        from SynthPolicyNet.data_utils import build_graph_from_smiles as _bg
                        g_child = _bg(child_smi)
                        if g_child is None:
                            from torch_geometric.data import Data as _TG
                            g_child = _TG(x=torch.zeros((1, node_dim), dtype=torch.float32), edge_index=torch.zeros((2, 0), dtype=torch.long))
                        if not hasattr(g_child, "batch"):
                            g_child.batch = torch.zeros((g_child.x.size(0),), dtype=torch.long)
                        g_child = g_child.to(device)
                        h_child = cond_policy.compute_h_state_block(g_child, protein_emb)
                        # Optional source conditioning
                        if bool(getattr(args, "pb_source_aware", False)):
                            src_id = 0  # dataset/internal transition
                            src_ids = torch.tensor([src_id], dtype=torch.long, device=device)
                            h_child = getattr(cond_policy, "compute_h_state_block_with_source")(g_child, protein_emb, src_ids)
                        # Block probability P_B(block | child)
                        block_logits_child = base_mod.compute_block_logits(h_child, block_embs).squeeze(0)
                        logp_b = F.log_softmax(block_logits_child, dim=-1)[b_idx]
                        # Rxn probability P_B(rxn | child, block)
                        sel_block = block_embs.index_select(0, torch.tensor([b_idx], dtype=torch.long, device=device))
                        rxn_input = torch.cat([h_child, sel_block], dim=1)
                        rxn_logits_child = base_mod.reaction_head(rxn_input).squeeze(0)
                        logp_r = F.log_softmax(rxn_logits_child, dim=-1)[r_idx]
                        log_pb_total = log_pb_total + (logp_b + logp_r)
                        state = child_smi
                    except Exception:
                        state = str(row_local.iloc[0]["result_smiles"]) or state
                else:
                    state = str(row_local.iloc[0]["result_smiles"]) or state

            # Terminal product reward and TB/DB/Sub‑TB loss
            reward = rc.get_reward(state, protein)
            # Capture Vina energies when available
            try:
                if hasattr(rc, "last_vina_energy") and rc.last_vina_energy is not None:
                    epoch_vina_min.append(float(rc.last_vina_energy))
                if hasattr(rc, "last_vina_raw_energy") and rc.last_vina_raw_energy is not None:
                    epoch_vina_raw.append(float(rc.last_vina_raw_energy))
            except Exception:
                pass
            # Auto-collect reference SMILES when Vina affinity is strong enough
            try:
                if auto_ref_th < 0.0 and getattr(rc, "use_vina", False):
                    ev = getattr(rc, "last_vina_energy", None)
                    if ev is not None and float(ev) < auto_ref_th and isinstance(state, str) and state:
                        m = Chem.MolFromSmiles(state)
                        if m is not None:
                            can = Chem.MolToSmiles(m)
                            if can and (can not in auto_ref_seen):
                                os.makedirs(os.path.dirname(auto_ref_out) or ".", exist_ok=True)
                                with open(auto_ref_out, "a", encoding="utf-8") as f:
                                    f.write(can + "\n")
                                auto_ref_seen.add(can)
                                print({"auto_ref_add": {"smiles": can, "vina_affinity_min": float(ev), "out": auto_ref_out}})
            except Exception:
                pass
            # Optional scaffold reward at finetune stage
            if bool(getattr(args, "use_scaffold_reward", False)) and str(getattr(args, "training_stage", "pretrain")) == "finetune":
                try:
                    sr = compute_scaffold_reward(state)
                    reward = float(reward) + float(getattr(args, "scaffold_weight", 0.2)) * float(sr)
                except Exception:
                    pass
            # Optional local shaping
            if bool(getattr(args, "use_local_reward", False)):
                try:
                    reward = float(reward) + float(getattr(args, "local_reward_weight", 0.1)) * float(local_accum)
                except Exception:
                    pass
            # Novelty shaping (terminal)
            nw = float(getattr(args, "novelty_weight", 0.0))
            if nw > 0.0 and isinstance(state, str) and state:
                try:
                    m = Chem.MolFromSmiles(state)
                    if m is not None:
                        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
                        # Lazy cache novelty fps in closure once
                        if not hasattr(main, "_novelty_fps"):
                            fps: list = []
                            files = list(getattr(args, "novelty_db", []))
                            for pth in files:
                                try:
                                    with open(pth, "r", encoding="utf-8", errors="ignore") as f:
                                        for line in f:
                                            s = line.strip()
                                            if not s:
                                                continue
                                            if "," in s:
                                                s = s.split(",")[0]
                                            mm = Chem.MolFromSmiles(s)
                                            if mm is None:
                                                continue
                                            fps.append(AllChem.GetMorganFingerprintAsBitVect(mm, radius=2, nBits=2048))
                                except Exception:
                                    continue
                            setattr(main, "_novelty_fps", fps)
                        ref_fps = getattr(main, "_novelty_fps", [])
                        max_sim = 0.0
                        for rf in ref_fps[:5000]:
                            try:
                                sim = float(DataStructs.TanimotoSimilarity(fp, rf))
                            except Exception:
                                sim = 0.0
                            if sim > max_sim:
                                max_sim = sim
                        reward = max(1e-8, float(reward) + nw * (1.0 - max_sim))
                except Exception:
                    pass
            # Collect metrics
            try:
                epoch_rewards.append(float(reward))
                if isinstance(state, str) and state:
                    epoch_terminals.add(state)
                    epoch_counts[state] += 1
            except Exception:
                pass
            # Success rate for this episode (only if at least one attempt)
            if attempted_steps > 0:
                ep_success = float(successful_steps) / float(attempted_steps)
                success_acc += ep_success
                success_n += 1

            # Accumulate open-space metrics
            epoch_template_steps += int(template_steps)
            epoch_freewalk_steps += int(freewalk_steps)
            epoch_dataset_success_steps += int(dataset_success_steps)

            # Optional pruning: skip gradient update for bad docking episodes
            prune_th = float(getattr(args, "prune_bad_vina_th", 0.0))
            do_prune = False
            try:
                if prune_th < 0.0 and getattr(rc, "use_vina", False):
                    if getattr(rc, "last_vina_energy", None) is not None:
                        do_prune = float(rc.last_vina_energy) > prune_th
                        if do_prune:
                            print({"pruned_episode": {"vina_affinity_min": float(rc.last_vina_energy), "threshold": prune_th}})
            except Exception:
                do_prune = False
            # Optional pruning: skip episodes with molecular weight above threshold
            if not do_prune:
                try:
                    mw_th = float(getattr(args, "prune_mw_th", 0.0))
                    if mw_th > 0.0 and isinstance(state, str) and state:
                        m_mw = Chem.MolFromSmiles(state)
                        if m_mw is not None:
                            mw = float(Descriptors.MolWt(m_mw))
                            if mw > mw_th:
                                do_prune = True
                                print({"pruned_episode": {"mw": mw, "threshold": mw_th}})
                except Exception:
                    pass
            if do_prune:
                # Skip TB loss and optimizer step for this episode
                continue

            # Backward policy contribution
            if bool(getattr(args, "pb_learned", False)):
                # If logsumexp marginalization is enabled, compute per-episode child-marginal logP_B(child)
                if bool(getattr(args, "pb_logsumexp", False)):
                    try:
                        # Build candidate parents for terminal state
                        child = state
                        cand_rows = fwd_df[fwd_df["result_smiles"] == child]
                        terms: List[torch.Tensor] = []
                        cap = max(1, int(getattr(args, "pb_candidate_cap", 64)))
                        if not cand_rows.empty:
                            # Sample up to cap rows
                            cand_rows = cand_rows.sample(n=min(cap, len(cand_rows)), random_state=42)
                            # Encode child once
                            from SynthPolicyNet.data_utils import build_graph_from_smiles as _bg
                            g_child = _bg(child)
                            if g_child is None:
                                from torch_geometric.data import Data as _TG
                                g_child = _TG(x=torch.zeros((1, node_dim), dtype=torch.float32), edge_index=torch.zeros((2, 0), dtype=torch.long))
                            if not hasattr(g_child, "batch"):
                                g_child.batch = torch.zeros((g_child.x.size(0),), dtype=torch.long)
                            g_child = g_child.to(device)
                            h_child = cond_policy.compute_h_state_block(g_child, protein_emb)
                            if bool(getattr(args, "pb_source_aware", False)):
                                src_ids = torch.tensor([0], dtype=torch.long, device=device)  # assume dataset source for terminal
                                h_child = getattr(cond_policy, "compute_h_state_block_with_source")(g_child, protein_emb, src_ids)
                            for _, r in cand_rows.iterrows():
                                b_smi = str(r.get("action_building_block", ""))
                                r_tpl = str(r.get("action_reaction_template", ""))
                                b_idx_c = dataset.block_vocab.stoi.get(b_smi, None)
                                r_idx_c = dataset.rxn_vocab.stoi.get(r_tpl, None)
                                if b_idx_c is None or r_idx_c is None:
                                    continue
                                # log P_B(b | child)
                                block_logits_child = base_mod.compute_block_logits(h_child, block_embs).squeeze(0)
                                logp_b = F.log_softmax(block_logits_child, dim=-1)[int(b_idx_c)]
                                # log P_B(r | child, b)
                                sel_block = block_embs.index_select(0, torch.tensor([int(b_idx_c)], dtype=torch.long, device=device))
                                rxn_input = torch.cat([h_child, sel_block], dim=1)
                                rxn_logits_child = base_mod.reaction_head(rxn_input).squeeze(0)
                                logp_r = F.log_softmax(rxn_logits_child, dim=-1)[int(r_idx_c)]
                                terms.append((logp_b + logp_r).unsqueeze(0))
                        else:
                            # No inbound edges in dataset; approximate with open top-k combinations
                            from SynthPolicyNet.data_utils import build_graph_from_smiles as _bg
                            g_child = _bg(child)
                            if g_child is None:
                                from torch_geometric.data import Data as _TG
                                g_child = _TG(x=torch.zeros((1, node_dim), dtype=torch.float32), edge_index=torch.zeros((2, 0), dtype=torch.long))
                            if not hasattr(g_child, "batch"):
                                g_child.batch = torch.zeros((g_child.x.size(0),), dtype=torch.long)
                            g_child = g_child.to(device)
                            h_child = cond_policy.compute_h_state_block(g_child, protein_emb)
                            if bool(getattr(args, "pb_source_aware", False)):
                                src_ids = torch.tensor([0], dtype=torch.long, device=device)
                                h_child = getattr(cond_policy, "compute_h_state_block_with_source")(g_child, protein_emb, src_ids)
                            # top-k blocks and reactions
                            b_logits = base_mod.compute_block_logits(h_child, block_embs).squeeze(0)
                            Kb = min(int(getattr(args, "pb_open_topk_block", 16)), b_logits.numel())
                            vb, ib = torch.topk(b_logits, k=Kb)
                            for j in range(Kb):
                                b_idx_c = int(ib[j].item())
                                sel_block = block_embs.index_select(0, torch.tensor([b_idx_c], dtype=torch.long, device=device))
                                rxn_input = torch.cat([h_child, sel_block], dim=1)
                                r_logits = base_mod.reaction_head(rxn_input).squeeze(0)
                                Kr = min(int(getattr(args, "pb_open_topk_rxn", 8)), r_logits.numel())
                                vr, ir = torch.topk(r_logits, k=Kr)
                                logp_b = F.log_softmax(b_logits, dim=-1)[b_idx_c]
                                logp_r_all = F.log_softmax(r_logits, dim=-1)
                                for k in range(Kr):
                                    r_idx_c = int(ir[k].item())
                                    terms.append((logp_b + logp_r_all[r_idx_c]).unsqueeze(0))
                        if terms:
                            log_pb = torch.logsumexp(torch.cat(terms, dim=0), dim=0)
                        else:
                            log_pb = log_pb_total  # fallback to per-step accumulation
                    except Exception:
                        log_pb = log_pb_total
                else:
                    log_pb = log_pb_total
            else:
                # Approx: inbound degree of terminal
                use_db = bool(getattr(args, "use_backward_policy", False))
                if use_db:
                    inbound = int(fwd_df[fwd_df["result_smiles"] == state].shape[0])
                    if inbound <= 0:
                        log_pb = torch.tensor(0.0, device=device)
                    else:
                        log_pb = torch.log(torch.tensor(1.0 / float(inbound), device=device))
                else:
                    log_pb = torch.tensor(0.0, device=device)

            residual = log_z + log_pf_total - torch.log(torch.tensor(max(1e-8, float(reward)), device=device)) - log_pb
            # Sub‑TB on trailing K steps (optional): approximate by scaling residual if K < full length
            sub_k = int(getattr(args, "sub_tb_k", 0))
            if sub_k > 0 and _step + 1 >= sub_k:
                # Heuristic: weight residual from last K steps higher (no exact per-step log prob tracked here)
                residual = residual * 0.7  # lighter constraint on full path, emphasize recent steps elsewhere
            # Residual clipping for stability
            clip_v = float(getattr(args, "tb_residual_clip", 10.0))
            residual = torch.clamp(residual, min=-clip_v, max=clip_v)
            loss = residual ** 2
            # Optional PB BC auxiliary: maximize log P_B(child) via marginalization (reuses the computed log_pb when learned)
            pb_bc_w = float(getattr(args, "pb_bc_weight", 0.0))
            if pb_bc_w > 0.0 and bool(getattr(args, "pb_learned", False)):
                loss = loss + pb_bc_w * (-log_pb)
            # Guard against NaN/Inf loss
            if not torch.isfinite(loss):
                continue
            optim.zero_grad(set_to_none=True)
            loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(cond_policy.parameters(), max_norm=1.0)
            except Exception:
                pass
            optim.step()
            total_loss += float(loss.detach().cpu().item())
            valid_loss_count += 1

        # Save checkpoint each epoch
        if rank == 0:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            base_state = getattr(getattr(cond_policy, "module", cond_policy), "base").state_dict()
            torch.save(
                {
                    "model_state": base_state,
                    "log_z": float(log_z.detach().cpu().item()),
                    "block_vocab": dataset.block_vocab.to_json(),
                    "rxn_vocab": dataset.rxn_vocab.to_json(),
                    "hidden_dim": int(hidden_dim),
                    "num_gnn_layers": int(num_layers),
                },
                args.save,
            )
            # Compute reward stats
            avg_r = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
            p50 = float(np.percentile(epoch_rewards, 50)) if epoch_rewards else 0.0
            p90 = float(np.percentile(epoch_rewards, 90)) if epoch_rewards else 0.0
            uniq = int(len(epoch_terminals))
            uniq_rate = float(uniq / max(1, len(epoch_rewards)))
            # Distribution collapse proxies
            N = max(1, len(epoch_rewards))
            if epoch_counts:
                # Only terminal counts (exclude special keys)
                term_counts = [int(v) for k, v in epoch_counts.items() if not str(k).startswith("__")]
                if term_counts:
                    top1_share = float(max(term_counts) / N)
                    # Herfindahl–Hirschman Index (sum of squared shares)
                    hhi = float(sum((c / N) ** 2 for c in term_counts))
                else:
                    top1_share = 0.0
                    hhi = 0.0
            else:
                top1_share = 0.0
                hhi = 0.0
            success_rate = (success_acc / max(1, success_n)) if success_n > 0 else 0.0
            executed_steps = max(1, epoch_dataset_success_steps + epoch_freewalk_steps)
            template_ratio = float(epoch_template_steps) / float(executed_steps)
            freewalk_ratio = float(epoch_freewalk_steps) / float(executed_steps)
            print({
                "epoch": epoch,
                "avg_tb_loss": total_loss / max(1, valid_loss_count),
                "avg_qsar_reward": avg_r,
                "p50_qsar": p50,
                "p90_qsar": p90,
                "unique_terminals": uniq,
                "unique_rate": uniq_rate,
                "top1_share": top1_share,
                "hhi": hhi,
                "success_rate": success_rate,
                "template_steps": int(epoch_template_steps),
                "template_ratio": template_ratio,
                "freewalk_steps": int(epoch_freewalk_steps),
                "freewalk_ratio": freewalk_ratio,
                "episodes": int(len(epoch_rewards)),
                "saved": args.save,
                "rank": rank,
                "world_size": world_size,
            })

            # Persist plotting-friendly metrics
            try:
                # Use absolute paths under project root to avoid CWD issues
                runs_dir = os.path.join(project_root, "runs")
                os.makedirs(runs_dir, exist_ok=True)
                # CSV (append with header on first write)
                csv_path = os.path.join(runs_dir, "online_tb_metrics.csv")
                write_header = not os.path.exists(csv_path)
                with open(csv_path, "a", encoding="utf-8") as f:
                    if write_header:
                        f.write(
                            "epoch,episodes,avg_tb_loss,avg_qsar_reward,p50_qsar,p90_qsar,unique_terminals,unique_rate,top1_share,hhi,success_rate,template_steps,template_ratio,freewalk_steps,freewalk_ratio\n"
                        )
                    f.write(
                        f"{epoch},{len(epoch_rewards)},{(total_loss / max(1, valid_loss_count)):.6f},{avg_r:.6f},{p50:.6f},{p90:.6f},{uniq},{uniq_rate:.6f},{top1_share:.6f},{hhi:.6f},{success_rate:.6f},{int(epoch_template_steps)},{template_ratio:.6f},{int(epoch_freewalk_steps)},{freewalk_ratio:.6f}\n"
                    )

                # New: per-episode Vina energies CSV (episode index within epoch)
                try:
                    if getattr(rc, "use_vina", False):
                        vinacsv = os.path.join(runs_dir, "online_tb_vina_episodes.csv")
                        write_header_v = not os.path.exists(vinacsv)
                        with open(vinacsv, "a", encoding="utf-8") as fv:
                            if write_header_v:
                                fv.write("epoch,episode_idx,vina_affinity_min,vina_affinity_raw\n")
                            # Align lengths safely (some episodes may lack Vina if fallback used)
                            n = max(len(epoch_vina_min), len(epoch_vina_raw), len(epoch_rewards))
                            for i in range(n):
                                vmin = epoch_vina_min[i] if i < len(epoch_vina_min) else ""
                                vraw = epoch_vina_raw[i] if i < len(epoch_vina_raw) else ""
                                fv.write(f"{epoch},{i},{vmin},{vraw}\n")
                        print({"vina_episodes_csv": vinacsv})
                except Exception:
                    pass

                # JSONL with terminal counts per epoch
                jsonl_path = os.path.join(runs_dir, "online_tb_terminals.jsonl")
                term_list = (
                    [{"smiles": s, "count": int(c)} for s, c in sorted(epoch_counts.items(), key=lambda kv: kv[1], reverse=True)]
                    if epoch_counts
                    else []
                )
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"epoch": epoch, "terminal_counts": term_list}, ensure_ascii=False) + "\n")
                print({"metrics_csv": csv_path, "terminals_jsonl": jsonl_path})
            except Exception:
                pass


if __name__ == "__main__":
    main()


