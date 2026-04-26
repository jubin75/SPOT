from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Batch
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from SynthPolicyNet.datasets import ForwardTrajectoryDataset
from SynthPolicyNet.data_utils import build_graph_from_smiles
from SynthPolicyNet.train_policy import build_forward_dataset
from SynthPolicyNet.models import SynthPolicyNet
from LeadGFlowNet.protein_encoder import SimpleProteinEncoder, Esm2ProteinEncoder, tokenize_protein
from LeadGFlowNet.conditional_policy import ConditionalSynthPolicy
from LeadGFlowNet.oracle import binarize_pactivity


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline Trajectory Balance training with pActivity")
    # Data
    p.add_argument("--input", default="data/reaction_paths_all_routes.csv", help="Retrosynthesis CSV path")
    p.add_argument("--forward", default="data/forward_trajectories.csv", help="Forward trajectories CSV path")
    p.add_argument("--rebuild-forward", action="store_true", help="Rebuild forward CSV even if exists")
    p.add_argument("--max-block-mw", type=float, default=200.0, help="Maximum molecular weight for building block reactants (Da)")
    p.add_argument("--pactivity", default="data/protein_ligand_pactivity.csv", help="Protein-ligand pActivity CSV")
    p.add_argument("--min-block-freq", type=int, default=1, help="Prune building blocks that appear fewer than this many times")
    p.add_argument("--pact-high", type=float, default=6.0)
    p.add_argument("--pact-low", type=float, default=5.0)
    p.add_argument("--include-negatives", action="store_true", help="Include negatives with small epsilon reward")
    p.add_argument("--neg-epsilon", type=float, default=1e-3, help="Reward value for negatives if included")
    # Model
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-gnn-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--share-encoders", action="store_true")
    p.add_argument("--use-l2-norm", action="store_true", help="Use cosine sim + temperature for block logits")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--checkpoint", type=str, default="checkpoints/synth_policy_net.pt")
    p.add_argument("--use-checkpoint", action="store_true")
    p.add_argument("--save", type=str, default="checkpoints/leadgflownet_offline_tb.pt")
    # Rxn-first option
    p.add_argument("--rxn-first", action="store_true", help="Use reaction-first factorization in computing log P_F")
    # Train
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-trajectories", type=int, default=2000, help="Cap number of trajectories for training")
    p.add_argument("--batch-size", type=int, default=8, help="Number of trajectories per TB batch")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--use-cpu", action="store_true")
    # Backward policy (PB) supervised pretraining epochs before TB
    p.add_argument("--pb-bc-pretrain-epochs", type=int, default=0, help="If >0, run PB maximum likelihood pretraining for these epochs before TB")
    # Protein encoder
    p.add_argument("--protein-encoder", type=str, default="simple", choices=["simple", "esm2"], help="Protein encoder backend")
    p.add_argument("--esm2-model", type=str, default="facebook/esm2_t30_150M_UR50D", help="HF model id for ESM2 when protein-encoder=esm2")
    # Distributed
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Select device")
    p.add_argument("--cuda-id", type=int, default=0)
    p.add_argument("--distributed", type=str, default="none", choices=["none", "ddp"], help="Enable DDP")
    p.add_argument("--dist-backend", type=str, default="nccl")
    p.add_argument("--dist-init-method", type=str, default="env://")
    return p


def select_longest_chain_indices(df: pd.DataFrame) -> List[int]:
    """Given a filtered df (single ligand optional), return indices of the longest chain for each (ligand, route_id).

    Assumes df has 'is_in_forward_chain' and 'forward_step_index'. Returns concatenated row indices sorted by step.
    """
    out_indices: List[int] = []
    if df.empty:
        return out_indices
    # Group by ligand+route and pick on-chain rows
    for (lig, rid), g in df.groupby(["ligand_smiles", "route_id"], sort=False):
        g_on = g[g.get("is_in_forward_chain", True).astype(bool)]
        if g_on.empty:
            continue
        # Sort by forward_step_index if exists, else step_index, else original order
        if "forward_step_index" in g_on.columns:
            g_on = g_on.sort_values(by=["forward_step_index"], kind="stable")
        elif "step_index" in g_on.columns:
            g_on = g_on.sort_values(by=["step_index"], kind="stable")
        out_indices.extend(g_on.index.tolist())
    return out_indices


def build_episodes(dataset: ForwardTrajectoryDataset, pact_df: pd.DataFrame, pact_high: float, pact_low: float,
                   include_negatives: bool, neg_epsilon: float) -> List[Dict]:
    """Join pActivity to forward chains by ligand and produce trajectories with labels/rewards.

    Returns a list of dicts with fields:
      - ligand_smiles
      - protein_sequence
      - reward (float > 0)
      - indices (List[int]) -> row indices into dataset.df representing the chain steps in order
    """
    df = dataset.df  # filtered inside dataset ctor
    # Build mapping ligand -> rows in longest chain (concatenated indices)
    episodes: List[Dict] = []
    # Ensure unique ligands in pActivity
    pact_df = pact_df.dropna(subset=["ligand_smiles", "protein_sequence", "p_activity"]).copy()
    for _, row in pact_df.iterrows():
        lig = str(row["ligand_smiles"])  # join key
        prot = str(row["protein_sequence"])
        y = binarize_pactivity(row["p_activity"], high=pact_high, low=pact_low)
        if y is None:
            continue  # ambiguous zone discarded
        reward = 1.0 if y == 1 else (float(neg_epsilon) if include_negatives else None)
        if reward is None or reward <= 0:
            continue
        g = df[df["ligand_smiles"] == lig]
        if g.empty:
            continue
        idxs = select_longest_chain_indices(g)
        if not idxs:
            continue
        episodes.append({
            "ligand_smiles": lig,
            "protein_sequence": prot,
            "reward": float(reward),
            "indices": idxs,
        })
    return episodes


def compute_log_pf_for_chain(dataset: ForwardTrajectoryDataset,
                             cond_policy: ConditionalSynthPolicy,
                             block_embs: torch.Tensor,
                             protein_emb_1x: torch.Tensor,
                             chain_indices: List[int],
                             device: torch.device) -> torch.Tensor:
    """Sum log P_F over steps of a chain.

    Returns a scalar tensor (on device).
    """
    # Build per-step Data list
    data_list = [dataset[i] for i in chain_indices]
    batch = Batch.from_data_list(data_list).to(device)
    B = batch.num_graphs
    prot_b = protein_emb_1x.expand(B, -1)
    # Determine mode from module attribute if available
    use_rxn_first = getattr(cond_policy, "use_rxn_first", False)
    if use_rxn_first:
        # Unconditional rxn on protein-conditioned state
        uncond_rxn_logits, _ = cond_policy.rxn_first(
            state_batch=batch,
            block_embeddings=block_embs,
            protein_emb=prot_b,
            rxn_indices_for_blocks=None,
        )
        # Teacher rxn for block logits (reaction-conditioned block selection)
        block_logits = cond_policy.base.compute_block_logits_given_rxn_h(
            cond_policy.compute_h_state_block(batch, prot_b), block_embs, batch.y_rxn
        )
        logp_rxn = F.log_softmax(uncond_rxn_logits, dim=1).gather(1, batch.y_rxn.view(-1, 1)).sum()
        logp_block = F.log_softmax(block_logits, dim=1).gather(1, batch.y_block.view(-1, 1)).sum()
        return logp_rxn + logp_block
    else:
        block_logits, rxn_logits = cond_policy(
            state_batch=batch,
            block_embeddings=block_embs,
            protein_emb=prot_b,
            block_indices_for_reaction=batch.y_block,
        )
        # Log-softmax and gather
        logp_block = F.log_softmax(block_logits, dim=1).gather(1, batch.y_block.view(-1, 1)).sum()
        logp_rxn = F.log_softmax(rxn_logits, dim=1).gather(1, batch.y_rxn.view(-1, 1)).sum()
        return logp_block + logp_rxn
def compute_log_pb_for_chain(dataset: ForwardTrajectoryDataset,
                             cond_policy: ConditionalSynthPolicy,
                             block_embs: torch.Tensor,
                             protein_emb_1x: torch.Tensor,
                             chain_indices: List[int],
                             device: torch.device) -> torch.Tensor:
    """Sum log P_B over steps of a chain using a child-conditioned factorization:

    P_B(block | child) * P_B(rxn | child, block)

    Child is the result_smiles of each transition in the chain.
    """
    from torch_geometric.data import Data as TGData
    # Build child graphs and gather labels per step
    child_graphs = []
    y_blocks: List[int] = []
    y_rxns: List[int] = []
    for idx in chain_indices:
        row = dataset.df.iloc[idx]
        child_smi = str(row.get("result_smiles", ""))
        g = build_graph_from_smiles(child_smi)
        if g is None:
            g = TGData(
                x=torch.zeros((1, dataset.node_feature_dim), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )
        child_graphs.append(g)
        d = dataset[idx]
        y_blocks.append(int(d.y_block.item()))
        y_rxns.append(int(d.y_rxn.item()))

    if not child_graphs:
        return torch.tensor(0.0, device=device)

    from torch_geometric.data import Batch
    batch_child = Batch.from_data_list(child_graphs).to(device)
    B = batch_child.num_graphs
    prot_b = protein_emb_1x.expand(B, -1)

    # Compute child-conditioned features
    h_child = cond_policy.compute_h_state_block(batch_child, prot_b)
    # Block logits over vocabulary
    block_logits = cond_policy.base.compute_block_logits(h_child, block_embs)
    yb = torch.tensor(y_blocks, dtype=torch.long, device=device)
    logp_block = F.log_softmax(block_logits, dim=1).gather(1, yb.view(-1, 1)).sum()

    # Reaction logits conditioned on selected block per step
    selected_block_embs = block_embs.index_select(0, yb)
    rxn_input = torch.cat([h_child, selected_block_embs], dim=1)
    rxn_logits = cond_policy.base.reaction_head(rxn_input)
    yr = torch.tensor(y_rxns, dtype=torch.long, device=device)
    logp_rxn = F.log_softmax(rxn_logits, dim=1).gather(1, yr.view(-1, 1)).sum()

    return logp_block + logp_rxn


def _get_rank_world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def main() -> None:
    args = build_argparser().parse_args()

    # Device
    dev_arg = (args.device or "auto").lower()
    if dev_arg == "cuda":
        device = torch.device(f"cuda:{args.cuda_id}")
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            torch.backends.cudnn.benchmark = True
    elif dev_arg == "mps":
        device = torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device("cpu")
    elif dev_arg == "cpu":
        device = torch.device("cpu")
    else:
        if not args.use_cpu and torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.backends.cudnn.benchmark = True
        elif not args.use_cpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if args.distributed == "ddp" and device.type == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", args.cuda_id))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_init_method)
    rank, world_size = _get_rank_world()
    is_main = (rank == 0)
    if is_main:
        print({"device": str(device), "rank": rank, "world_size": world_size})

    # Forward data and dataset (provides vocab + block graphs)
    fwd_df = build_forward_dataset(
        args.input,
        args.forward,
        skip_start_steps=True,
        rebuild=args.rebuild_forward,
        max_block_mw=args.max_block_mw,
    )
    dataset = ForwardTrajectoryDataset(
        fwd_df,
        block_vocab=None,
        rxn_vocab=None,
        use_only_forward_chain=True,
        skip_start_states=True,
        min_block_freq=max(1, int(args.min_block_freq)),
    )
    if is_main:
        print({
            "samples": len(dataset),
            "num_blocks": len(dataset.block_vocab.itos),
            "num_rxns": len(dataset.rxn_vocab.itos),
            "node_feat_dim": dataset.node_feature_dim,
        })

    # Build base model and conditional wrapper
    # Infer dims from checkpoint if requested/available to avoid size mismatch
    inferred_hidden = int(args.hidden_dim)
    inferred_layers = int(args.num_gnn_layers)
    ckpt_obj = None
    if args.use_checkpoint and os.path.exists(args.checkpoint):
        try:
            ckpt_obj = torch.load(args.checkpoint, map_location=device)
            if isinstance(ckpt_obj, dict):
                if "hidden_dim" in ckpt_obj:
                    inferred_hidden = int(ckpt_obj["hidden_dim"])
                else:
                    sd0 = ckpt_obj.get("model_state", ckpt_obj)
                    w = sd0.get("state_encoder.convs.0.lin.weight") if isinstance(sd0, dict) else None
                    if w is not None and hasattr(w, "shape"):
                        inferred_hidden = int(w.shape[0])
                if "num_gnn_layers" in ckpt_obj:
                    inferred_layers = int(ckpt_obj["num_gnn_layers"])
                else:
                    sd0 = ckpt_obj.get("model_state", ckpt_obj)
                    if isinstance(sd0, dict):
                        layer_count = sum(1 for k in sd0.keys() if k.startswith("state_encoder.convs.") and k.endswith(".lin.weight"))
                        if layer_count > 0:
                            inferred_layers = int(layer_count)
        except Exception:
            ckpt_obj = None

    base = SynthPolicyNet(
        node_feature_dim=dataset.node_feature_dim,
        hidden_dim=int(inferred_hidden),
        num_building_blocks=len(dataset.block_vocab.itos),
        num_reaction_templates=len(dataset.rxn_vocab.itos),
        num_gnn_layers=int(inferred_layers),
        dropout=float(args.dropout),
        share_encoders=bool(args.share_encoders),
        use_l2_normalization=bool(args.use_l2_norm),
        initial_temperature=float(args.temperature),
    ).to(device)
    if ckpt_obj is not None:
        # Shape-safe loading: only load params whose shapes match current model
        try:
            src_sd = ckpt_obj.get("model_state", ckpt_obj)
            dst_sd = base.state_dict()
            filtered = {}
            skipped = []
            for k, v in (src_sd.items() if isinstance(src_sd, dict) else []):
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
            # Fallback: best-effort loading (may still skip mismatched under strict=False)
            base.load_state_dict(ckpt_obj.get("model_state", ckpt_obj), strict=False)
        print({"loaded_checkpoint": args.checkpoint, "hidden_dim": int(inferred_hidden), "num_gnn_layers": int(inferred_layers)})

    # Protein encoder
    use_esm = (args.protein_encoder == "esm2")
    if use_esm:
        prot_enc = Esm2ProteinEncoder(model_name=args.esm2_model).to(device)
        protein_dim = int(prot_enc.out_dim)
    else:
        prot_enc = SimpleProteinEncoder(embed_dim=args.hidden_dim // 2, lstm_hidden=args.hidden_dim // 2).to(device)
        protein_dim = int(args.hidden_dim)

    cond_policy = ConditionalSynthPolicy(base, protein_dim=protein_dim).to(device)
    # Flag for rxn-first usage in this script path
    setattr(cond_policy, "use_rxn_first", bool(args.rxn_first))
    if args.distributed == "ddp" and device.type == "cuda":
        cond_policy = DDP(cond_policy, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
    log_z = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))

    # Precompute block embeddings each epoch (weights change, so refresh)
    valid_block_graphs = []
    for g in dataset.block_graphs:
        if g is None:
            from torch_geometric.data import Data as TGData
            valid_block_graphs.append(
                TGData(
                    x=torch.zeros((1, dataset.node_feature_dim), dtype=torch.float32),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                )
            )
        else:
            valid_block_graphs.append(g)

    # Load pActivity and build episodes
    pact_df = pd.read_csv(args.pactivity)
    episodes_all = build_episodes(
        dataset,
        pact_df,
        pact_high=args.pact_high,
        pact_low=args.pact_low,
        include_negatives=args.include_negatives,
        neg_epsilon=args.neg_epsilon,
    )
    if args.max_trajectories and len(episodes_all) > args.max_trajectories:
        episodes_all = episodes_all[: args.max_trajectories]
    print({"episodes": len(episodes_all)})
    if not episodes_all:
        print({"error": "No episodes constructed. Check joins and thresholds."})
        return

    # Optimizer
    params = list(cond_policy.parameters()) + [log_z]
    optim = Adam(params, lr=args.lr)

    # Training loop
    # Optional PB BC pretraining (maximize log P_B on dataset chains)
    pb_bc_epochs = int(getattr(args, "pb_bc_pretrain_epochs", 0))
    for e in range(1, pb_bc_epochs + 1):
        base.train()
        base_mod = cond_policy.module.base if isinstance(cond_policy, DDP) else cond_policy.base
        block_embs = base_mod.encode_blocks(valid_block_graphs, device=device)
        total_ll = 0.0
        count_ep = 0
        for start in range(0, len(episodes_all), args.batch_size):
            batch_eps = episodes_all[start: start + args.batch_size]
            if not batch_eps:
                continue
            optim.zero_grad(set_to_none=True)
            loss_bc = 0.0
            for ep in batch_eps:
                if use_esm:
                    prot_emb_1x = prot_enc.encode_sequence(ep["protein_sequence"], device=device)
                else:
                    prot_ids = tokenize_protein(ep["protein_sequence"]).to(device)
                    prot_emb_1x = prot_enc(prot_ids)
                log_pb = compute_log_pb_for_chain(dataset, cond_policy, block_embs, prot_emb_1x, ep["indices"], device)
                # Maximize log_pb -> minimize negative log likelihood
                loss_bc = loss_bc - log_pb
                total_ll += float(log_pb.detach().cpu().item())
                count_ep += 1
            loss_bc = loss_bc / max(1, len(batch_eps))
            loss_bc.backward()
            optim.step()
        if is_main:
            avg_ll = total_ll / max(1, count_ep)
            print({"pb_bc_epoch": e, "avg_log_pb": avg_ll})

    for epoch in range(1, args.epochs + 1):
        base.train()
        # Refresh block embeddings
        base_mod = cond_policy.module.base if isinstance(cond_policy, DDP) else cond_policy.base
        block_embs = base_mod.encode_blocks(valid_block_graphs, device=device)
        total_loss = 0.0
        count = 0

        # Mini-batch over trajectories
        for start in range(0, len(episodes_all), args.batch_size):
            batch_eps = episodes_all[start: start + args.batch_size]
            if not batch_eps:
                continue
            optim.zero_grad(set_to_none=True)

            loss_batch = 0.0
            for ep in batch_eps:
                if use_esm:
                    prot_emb_1x = prot_enc.encode_sequence(ep["protein_sequence"], device=device)
                else:
                    prot_ids = tokenize_protein(ep["protein_sequence"]).to(device)
                    prot_emb_1x = prot_enc(prot_ids)  # (1, D)
                log_pf = compute_log_pf_for_chain(dataset, cond_policy, block_embs, prot_emb_1x, ep["indices"], device)
                # Learned backward log-prob (child-conditioned)
                log_pb = compute_log_pb_for_chain(dataset, cond_policy, block_embs, prot_emb_1x, ep["indices"], device)
                # Reward
                R = max(1e-8, float(ep["reward"]))
                log_r = torch.log(torch.tensor(R, device=device, dtype=torch.float32))
                tb = (log_z + log_pf - log_r - log_pb) ** 2
                loss_batch = loss_batch + tb

            # Normalize by number of episodes in this batch
            loss_batch = loss_batch / max(1, len(batch_eps))
            loss_batch.backward()
            optim.step()

            bs = len(batch_eps)
            total_loss += float(loss_batch.item()) * bs
            count += bs

        avg_loss = total_loss / max(1, count)
        if is_main:
            print({"epoch": epoch, "tb_loss": avg_loss, "logZ": float(log_z.detach().cpu().item())})

        # Save checkpoint each epoch
        if is_main:
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            base_state = (cond_policy.module.base.state_dict() if isinstance(cond_policy, DDP) else cond_policy.base.state_dict())
            torch.save(
                {
                    "model_state": base_state,
                    "log_z": float(log_z.detach().cpu().item()),
                    "block_vocab": dataset.block_vocab.to_json(),
                    "rxn_vocab": dataset.rxn_vocab.to_json(),
                },
                args.save,
            )
            print({"saved": args.save})


if __name__ == "__main__":
    main()


