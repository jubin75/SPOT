#!/usr/bin/env python3
import os
import json
import argparse
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt


def read_metrics_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"metrics CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Ensure expected columns exist
    expected = {
        "epoch",
        "episodes",
        "avg_tb_loss",
        "avg_qsar_reward",
        "p50_qsar",
        "p90_qsar",
        "unique_terminals",
        "unique_rate",
        "top1_share",
        "hhi",
        "success_rate",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"metrics CSV missing columns: {missing}")
    return df
def read_vina_episodes_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"vina episodes CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Expected columns: epoch, episode_idx, vina_affinity_min, vina_affinity_raw
    need = {"epoch", "episode_idx", "vina_affinity_min", "vina_affinity_raw"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"vina episodes CSV missing columns: {missing}")
    return df



def read_terminals_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"terminals JSONL not found: {jsonl_path}")
    out: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(obj)
    return out


def plot_time_series(df: pd.DataFrame, out_dir: str, vina_df: pd.DataFrame | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # 1) QSAR rewards
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["avg_qsar_reward"], label="avg_qsar_reward")
    ax.plot(df["epoch"], df["p50_qsar"], label="p50_qsar")
    ax.plot(df["epoch"], df["p90_qsar"], label="p90_qsar")
    ax.set_xlabel("epoch")
    ax.set_ylabel("QSAR reward")
    ax.set_title("QSAR reward vs epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "qsar_rewards.png"), dpi=200)
    plt.close(fig)

    # 2) Diversity metrics
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["unique_terminals"], label="unique_terminals")
    ax2 = ax.twinx()
    ax2.plot(df["epoch"], df["unique_rate"], color="tab:orange", label="unique_rate")
    ax.set_xlabel("epoch")
    ax.set_ylabel("unique terminals")
    ax2.set_ylabel("unique rate")
    ax.set_title("Diversity vs epoch")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "diversity.png"), dpi=200)
    plt.close(fig)

    # 3) Collapse proxies + success rate
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["top1_share"], label="top1_share")
    ax.plot(df["epoch"], df["hhi"], label="hhi")
    ax.plot(df["epoch"], df["success_rate"], label="success_rate")
    ax.set_xlabel("epoch")
    ax.set_ylabel("share / index / rate")
    ax.set_title("Collapse proxies & success vs epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "collapse_proxies.png"), dpi=200)
    plt.close(fig)

    # 4) TB loss
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["avg_tb_loss"], label="avg_tb_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("TB loss")
    ax.set_title("TB loss vs epoch")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tb_loss.png"), dpi=200)
    plt.close(fig)

    # 5) Vina affinity vs episodes (if provided)
    if vina_df is not None and not vina_df.empty:
        try:
            # Plot per-epoch series: concatenate and plot episode index across epochs
            # Build a continuous episode counter across all epochs
            vina_df_sorted = vina_df.sort_values(["epoch", "episode_idx"]).reset_index(drop=True)
            vina_df_sorted["global_ep"] = range(1, len(vina_df_sorted) + 1)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(vina_df_sorted["global_ep"], vina_df_sorted["vina_affinity_min"], label="vina_affinity_min", linewidth=1)
            ax.set_xlabel("episodes (cumulative)")
            ax.set_ylabel("Vina affinity (kcal/mol; lower is better)")
            ax.set_title("Vina affinity (min) vs episodes")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "vina_affinity_vs_episodes.png"), dpi=200)
            plt.close(fig)
        except Exception:
            pass


def plot_topk_terminals(terminals: List[Dict[str, Any]], out_dir: str, top_k: int = 20) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Pick the last epoch
    if not terminals:
        return
    last = max(terminals, key=lambda r: r.get("epoch", 0))
    counts = last.get("terminal_counts", [])
    counts = sorted(counts, key=lambda x: x.get("count", 0), reverse=True)[:top_k]
    if not counts:
        return
    labels = [c["smiles"] for c in counts]
    values = [c["count"] for c in counts]
    fig, ax = plt.subplots(figsize=(10, max(4, int(top_k * 0.3))))
    ax.barh(range(len(labels)), values)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("count")
    ax.set_title(f"Top-{len(labels)} terminal products (last epoch {last.get('epoch')})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "topk_terminals_last_epoch.png"), dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot online TB metrics and terminal distributions")
    p.add_argument("--metrics", default="runs/online_tb_metrics.csv")
    p.add_argument("--terminals", default="runs/online_tb_terminals.jsonl")
    p.add_argument("--out", default="runs/plots_online_tb")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--vina-episodes", default="runs/online_tb_vina_episodes.csv")
    args = p.parse_args()

    df = read_metrics_csv(args.metrics)
    terms = read_terminals_jsonl(args.terminals)

    # Optional: load Vina per-episode series
    vina_df = None
    try:
        vina_df = read_vina_episodes_csv(args.vina_episodes)
    except Exception:
        vina_df = None
    plot_time_series(df, args.out, vina_df=vina_df)
    plot_topk_terminals(terms, args.out, top_k=args.topk)
    print({"saved_dir": args.out})


if __name__ == "__main__":
    main()


