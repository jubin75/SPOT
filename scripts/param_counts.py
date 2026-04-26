from __future__ import annotations

import os
import json
from typing import Dict

import torch

from SynthPolicyNet.train_policy import build_forward_dataset
from SynthPolicyNet.datasets import ForwardTrajectoryDataset
from SynthPolicyNet.models import SynthPolicyNet
from LeadGFlowNet.conditional_policy import ConditionalSynthPolicy


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def find_project_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def find_esm2_hidden_size(default: int = 640) -> int:
    """Best-effort: read hidden_size from local ESM2 snapshot config.json if present."""
    root = find_project_root()
    snap_root = os.path.join(root, "lib", "models--facebook--esm2_t30_150M_UR50D", "snapshots")
    try:
        if not os.path.isdir(snap_root):
            return default
        snaps = [os.path.join(snap_root, d) for d in os.listdir(snap_root)]
        snaps = [p for p in snaps if os.path.isdir(p)]
        if not snaps:
            return default
        # Pick most recent snapshot
        snaps.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        cfg = os.path.join(snaps[0], "config.json")
        if os.path.isfile(cfg):
            with open(cfg, "r", encoding="utf-8") as f:
                obj = json.load(f)
            hs = int(obj.get("hidden_size", default))
            return hs
    except Exception:
        pass
    return default


def main() -> None:
    root = find_project_root()

    input_csv = os.path.join(root, "data", "reaction_paths_all_routes.csv")
    forward_csv = os.path.join(root, "data", "forward_trajectories.csv")

    # Ensure forward trajectories exist (no rebuild)
    fwd_df = build_forward_dataset(input_csv, forward_csv, skip_start_steps=True, rebuild=False, max_block_mw=200.0)
    dataset = ForwardTrajectoryDataset(fwd_df)

    hidden_dim = 256
    num_layers = 3

    base = SynthPolicyNet(
        node_feature_dim=dataset.node_feature_dim,
        hidden_dim=hidden_dim,
        num_building_blocks=len(dataset.block_vocab.itos),
        num_reaction_templates=len(dataset.rxn_vocab.itos),
        num_gnn_layers=num_layers,
        dropout=0.1,
        share_encoders=False,
        use_l2_normalization=True,
        initial_temperature=0.07,
    )

    synthpolicynet_params = count_params(base)

    # Offline TB: trainable params == ConditionalSynthPolicy(base, protein_dim=hidden_dim) + logZ
    cond_offline = ConditionalSynthPolicy(base, protein_dim=hidden_dim)
    offline_tb_trainable = count_params(cond_offline) + 1  # +1 for logZ scalar

    # Online TB: uses ESM2 (not trainable), but FiLM MLPs input dim changes
    esm2_hidden = find_esm2_hidden_size(default=640)
    cond_online = ConditionalSynthPolicy(base, protein_dim=esm2_hidden)
    online_tb_trainable = count_params(cond_online) + 1  # +1 for logZ scalar

    out: Dict[str, int] = {
        "SynthPolicyNet": int(synthpolicynet_params),
        "OfflineTB_trainable": int(offline_tb_trainable),
        "OnlineTB_trainable": int(online_tb_trainable),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


