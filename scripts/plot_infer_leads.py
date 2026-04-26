#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import QED, Crippen, Lipinski


def calc_props(smiles: str) -> Tuple[bool, float, float, float, int]:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False, 0.0, 10.0, 0.0, 4
    # QED
    try:
        qed = float(QED.qed(m))
    except Exception:
        qed = 0.0
    # SA proxy consistent with oracle heuristic
    try:
        heavy = m.GetNumHeavyAtoms()
        ri = m.GetRingInfo()
        num_rings = len(ri.AtomRings()) if ri is not None else 0
        hetero = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() not in (6, 1))
        sa = float(min(10.0, max(1.0, 1.0 + 0.1 * heavy + 0.3 * num_rings + 0.2 * hetero)))
    except Exception:
        sa = 10.0
    # MW and Lipinski violations
    try:
        from rdkit.Chem import Descriptors  # type: ignore
        mw = float(Descriptors.MolWt(m))
    except Exception:
        mw = float(Crippen.MolMR(m)) * 10.0
    logp = float(Crippen.MolLogP(m))
    hbd = int(Lipinski.NumHDonors(m))
    hba = int(Lipinski.NumHAcceptors(m))
    lip_viol = int((mw > 500) + (logp > 5) + (hbd > 5) + (hba > 10))
    return True, qed, sa, mw, lip_viol


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize online TB inference leads (JSON -> CSV + plots)")
    p.add_argument("--json", default="runs/lead_routes.json", help="Path to lead_routes.json from infer.sh")
    p.add_argument("--out", default="runs/plots_infer", help="Output directory for plots and CSV")
    args = p.parse_args()

    if not os.path.exists(args.json):
        raise FileNotFoundError(f"JSON not found: {args.json}")
    with open(args.json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    leads: List[str] = list(obj.get("leads_set", []))
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for s in leads:
        if isinstance(s, str) and s and (s not in seen):
            seen.add(s)
            uniq.append(s)

    rows = []
    valid_count = 0
    for s in uniq:
        ok, qed, sa, mw, lip = calc_props(s)
        valid_count += int(ok)
        rows.append({
            "smiles": s,
            "valid": int(ok),
            "qed": qed,
            "sa": sa,
            "mw": mw,
            "lip_viol": lip,
        })

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    csv_out = os.path.join(args.out, "leads_metrics.csv")
    df.to_csv(csv_out, index=False)
    print({
        "json": args.json,
        "leads": len(leads),
        "unique": len(uniq),
        "valid": int(valid_count),
        "csv": csv_out,
    })

    # Plots
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Helper: smooth density line for continuous metrics
        def plot_hist_with_envelope(values, rng=None, bins=60, color="#4472c4", title=""):
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                return
            plt.figure(figsize=(6, 4))
            # Histogram (density)
            plt.hist(values, bins=bins, density=True, alpha=0.5, color=color, edgecolor="none")
            # Smoothed envelope via Gaussian kernel on histogram density
            vmin = float(np.min(values))
            vmax = float(np.max(values))
            if rng is not None:
                vmin, vmax = float(rng[0]), float(rng[1])
            x_grid = np.linspace(vmin, vmax, 512)
            hist_counts, bin_edges = np.histogram(values, bins=bins, range=(vmin, vmax), density=True)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            # Build Gaussian kernel
            sigma = max(1, int(0.03 * bins))
            kx = np.arange(-3 * sigma, 3 * sigma + 1)
            kernel = np.exp(-(kx ** 2) / (2.0 * sigma ** 2))
            kernel = kernel / np.sum(kernel)
            smooth = np.convolve(hist_counts, kernel, mode="same")
            # Interpolate to x_grid for a smooth envelope
            smooth_y = np.interp(x_grid, bin_centers, smooth)
            plt.plot(x_grid, smooth_y, color=color, linewidth=2.0)
            plt.title(title)
            if rng is not None:
                plt.xlim(rng)
            plt.tight_layout()
            return plt

        # Continuous metrics with envelope: QED, SA, MW
        plot_hist_with_envelope(df["qed"].values, rng=(0.0, 1.0), title="qed")
        plt.savefig(os.path.join(args.out, "qed.png"), dpi=150)
        plt.close()

        plot_hist_with_envelope(df["sa"].values, rng=(1.0, 10.0), title="sa")
        plt.savefig(os.path.join(args.out, "sa.png"), dpi=150)
        plt.close()

        plot_hist_with_envelope(df["mw"].values, rng=None, title="mw")
        plt.savefig(os.path.join(args.out, "mw.png"), dpi=150)
        plt.close()

        # Discrete metric with percentage annotations: Lipinski violations
        vals = df["lip_viol"].astype(int).values
        total = max(1, vals.size)
        cats = np.arange(0, 5)
        counts = np.array([(vals == c).sum() for c in cats], dtype=float)
        perc = counts / float(total) * 100.0
        plt.figure(figsize=(6, 4))
        bars = plt.bar(cats, counts, color="#4472c4", alpha=0.9)
        plt.xticks(cats, [str(c) for c in cats])
        plt.xlim(-0.5, 4.5)
        plt.title("lip_viol (0 = no violation)")
        # Annotate percentage on each bar
        for rect, p in zip(bars, perc):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height + max(1.0, 0.01 * total), f"{p:.1f}%", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "lip_viol.png"), dpi=150)
        plt.close()
    except Exception as e:
        print({"plot_error": str(e)})


if __name__ == "__main__":
    main()


