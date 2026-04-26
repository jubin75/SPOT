#!/usr/bin/env python
from __future__ import annotations

"""
Evaluate Offline TB and Online TB models on the PDBBind-derived test set.

Inputs:
- data/pdbbind_testset.csv (from build_pdbbind_testset.py)
- checkpoints:
  - Offline TB: checkpoints/leadgflownet_offline_tb.pt (or synth_policy_net.pt as fallback)
  - Online TB:  checkpoints/leadgflownet_online_tb.pt (if present)
- forward/reaction CSVs for feasible graph

Outputs (under runs/pdbbind_eval/):
- CSV with similarity vs native ligands per protein (offline vs online)
- Plots: validity, uniqueness, novelty, QED, SA, Lipinski violations, similarity distributions (ECFP4 Tanimoto)
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, QED, Crippen, Lipinski
from rdkit import DataStructs

from leadgflownet_infer import main as infer_main


def ecfp4_fp(smiles: str, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.int8)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return float(inter / union) if union > 0 else 0.0


def qed_sa_lip(smiles: str) -> Tuple[float, float, int]:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return 0.0, 10.0, 4
    q = float(QED.qed(m))
    # SA proxy as in oracle
    heavy = m.GetNumHeavyAtoms()
    ri = m.GetRingInfo()
    num_rings = len(ri.AtomRings()) if ri is not None else 0
    hetero = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() not in (6, 1))
    sa = float(min(10.0, max(1.0, 1.0 + 0.1*heavy + 0.3*num_rings + 0.2*hetero)))
    # Lipinski violations
    mw = float(Crippen.MolMR(m)) * 10.0  # rough proxy; if Descriptors.MolWt not imported
    try:
        from rdkit.Chem import Descriptors
        mw = float(Descriptors.MolWt(m))
    except Exception:
        pass
    logp = float(Crippen.MolLogP(m))
    hbd = int(Lipinski.NumHDonors(m))
    hba = int(Lipinski.NumHAcceptors(m))
    viol = int((mw > 500) + (logp > 5) + (hbd > 5) + (hba > 10))
    return q, sa, viol


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate TB models on PDBBind test set")
    p.add_argument("--testset", default="data/pdbbind_testset.csv")
    p.add_argument("--input", default="data/reaction_paths_all_routes.csv")
    p.add_argument("--forward", default="data/forward_trajectories.csv")
    p.add_argument("--offline-ckpt", default="checkpoints/leadgflownet_offline_tb.pt")
    p.add_argument("--online-ckpt", default="checkpoints/leadgflownet_online_tb.pt")
    p.add_argument("--qsar-ckpt", default="checkpoints/qsar.pt")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--outdir", default="runs/pdbbind_eval")
    # Optional external baselines: CSVs with columns [pdb_id, smiles]
    p.add_argument("--baseline-files", nargs="*", default=[], help="CSV files of baseline generations with columns [pdb_id, smiles]")
    p.add_argument("--baseline-labels", nargs="*", default=[], help="Labels matching --baseline-files in order")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.testset)
    if df.empty:
        raise RuntimeError("Empty test set.")

    records: List[Dict[str, object]] = []
    for i, row in df.iterrows():
        prot = str(row["protein_sequence"])
        pdb_id = str(row["pdb_id"]) if "pdb_id" in row else f"PDB_{i:05d}"
        native_ligs = [s for s in str(row["ligand_smiles_list"]).split("|") if s.strip()]
        native_ligs = [s.strip() for s in native_ligs]
        # Precompute native fingerprints
        native_fps = [ecfp4_fp(s) for s in native_ligs]

        # Offline inference (fallback to BC if offline ckpt missing)
        offline_ckpt = args.offline_ckpt if os.path.exists(args.offline_ckpt) else "checkpoints/synth_policy_net.pt"
        os.system(
            f"python -u leadgflownet_infer.py --input {args.input} --forward {args.forward} "
            f"--checkpoint {offline_ckpt} --protein-encoder esm2 --esm2-model lib/models--facebook--esm2_t30_150M_UR50D "
            f"--protein '{prot}' --num-samples {args.num_samples} --max-depth {args.max_depth} "
            f"--branch-block-topk 2 --branch-rxn-topk 1 --temperature 1.0 --deterministic --output-json {os.path.join(args.outdir, 'offline.json')}"
        )
        # Online inference (prefer online ckpt; QSAR-guided as option)
        online_ckpt = args.online_ckpt if os.path.exists(args.online_ckpt) else offline_ckpt
        os.system(
            f"python -u leadgflownet_infer.py --input {args.input} --forward {args.forward} "
            f"--checkpoint {online_ckpt} --protein-encoder esm2 --esm2-model lib/models--facebook--esm2_t30_150M_UR50D "
            f"--protein '{prot}' --num-samples {args.num_samples} --max-depth {args.max_depth} "
            f"--branch-block-topk 2 --branch-rxn-topk 1 --temperature 1.0 --use-qsar --qsar-checkpoint {args.qsar_ckpt} --qsar-mix 0.7 "
            f"--deterministic --output-json {os.path.join(args.outdir, 'online.json')}"
        )

        # Load generated leads
        import json
        def load_leads(path: str) -> List[str]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    return list(obj.get("leads_set", []))
            except Exception:
                return []

        leads_off = load_leads(os.path.join(args.outdir, "offline.json"))
        leads_on = load_leads(os.path.join(args.outdir, "online.json"))
        # Compute metrics
        for tag, leads in [("offline", leads_off), ("online", leads_on)]:
            uniq_leads = list(dict.fromkeys([s for s in leads if s]))
            valids = [s for s in uniq_leads if Chem.MolFromSmiles(s) is not None]
            validity = float(len(valids) / max(1, len(uniq_leads)))
            uniqueness = float(len(set(valids)) / max(1, len(valids)))
            # Novelty vs native ligands (fraction with tanimoto < 0.4)
            fps_leads = [ecfp4_fp(s) for s in valids]
            sim_max = []
            for fp in fps_leads:
                max_sim = max((tanimoto(fp, nfp) for nfp in native_fps), default=0.0)
                sim_max.append(max_sim)
            novelty = float(np.mean([1.0 if x < 0.4 else 0.0 for x in sim_max])) if sim_max else 0.0
            # Similarity distribution summary
            sim_p50 = float(np.percentile(sim_max, 50)) if sim_max else 0.0
            sim_p90 = float(np.percentile(sim_max, 90)) if sim_max else 0.0
            # Medicinal chemistry metrics
            qeds, sas, lips = [], [], []
            for s in valids:
                q, sa, lip = qed_sa_lip(s)
                qeds.append(q); sas.append(sa); lips.append(lip)
            rec = {
                "pdb_id": pdb_id,
                "mode": tag,
                "num_generated": len(leads),
                "num_valid": len(valids),
                "validity": validity,
                "uniqueness": uniqueness,
                "novelty_lt0.4": novelty,
                "sim_p50": sim_p50,
                "sim_p90": sim_p90,
                "qed_mean": float(np.mean(qeds)) if qeds else 0.0,
                "sa_mean": float(np.mean(sas)) if sas else 10.0,
                "lip_viol_mean": float(np.mean(lips)) if lips else 4.0,
            }
            records.append(rec)

    # Integrate external baselines
    if args.baseline_files:
        labels = args.baseline_labels if args.baseline_labels and len(args.baseline_labels) == len(args.baseline_files) else [f"baseline_{i}" for i in range(len(args.baseline_files))]
        for path, label in zip(args.baseline_files, labels):
            try:
                bdf = pd.read_csv(path)
            except Exception:
                continue
            if not {"pdb_id", "smiles"}.issubset(set(bdf.columns)):
                continue
            # group by pdb_id
            for pdb_id, g in bdf.groupby("pdb_id", sort=False):
                natives = df[df["pdb_id"] == pdb_id]
                if natives.empty:
                    continue
                native_ligs = [s.strip() for s in str(natives.iloc[0]["ligand_smiles_list"]).split("|") if s.strip()]
                native_fps = [ecfp4_fp(s) for s in native_ligs]
                leads = [str(s) for s in g["smiles"].astype(str).tolist() if s]
                uniq = list(dict.fromkeys(leads))
                valids = [s for s in uniq if Chem.MolFromSmiles(s) is not None]
                validity = float(len(valids) / max(1, len(uniq)))
                uniqueness = float(len(set(valids)) / max(1, len(valids)))
                fps_leads = [ecfp4_fp(s) for s in valids]
                sim_max = []
                for fp in fps_leads:
                    max_sim = max((tanimoto(fp, nfp) for nfp in native_fps), default=0.0)
                    sim_max.append(max_sim)
                novelty = float(np.mean([1.0 if x < 0.4 else 0.0 for x in sim_max])) if sim_max else 0.0
                sim_p50 = float(np.percentile(sim_max, 50)) if sim_max else 0.0
                sim_p90 = float(np.percentile(sim_max, 90)) if sim_max else 0.0
                qeds, sas, lips = [], [], []
                for s in valids:
                    q, sa, lip = qed_sa_lip(s)
                    qeds.append(q); sas.append(sa); lips.append(lip)
                records.append({
                    "pdb_id": pdb_id,
                    "mode": label,
                    "num_generated": len(leads),
                    "num_valid": len(valids),
                    "validity": validity,
                    "uniqueness": uniqueness,
                    "novelty_lt0.4": novelty,
                    "sim_p50": sim_p50,
                    "sim_p90": sim_p90,
                    "qed_mean": float(np.mean(qeds)) if qeds else 0.0,
                    "sa_mean": float(np.mean(sas)) if sas else 10.0,
                    "lip_viol_mean": float(np.mean(lips)) if lips else 4.0,
                })

    out_csv = os.path.join(args.outdir, "pdbbind_eval_metrics.csv")
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print({"saved_metrics": out_csv})

    # Plot simple distributions
    try:
        import matplotlib.pyplot as plt
        dfm = pd.DataFrame(records)
        import seaborn as sns
        dfm = pd.DataFrame(records)
        modes = dfm["mode"].unique().tolist()
        for metric in ["validity", "uniqueness", "novelty_lt0.4", "sim_p50", "sim_p90", "qed_mean", "sa_mean", "lip_viol_mean"]:
            # Histogram overlay
            plt.figure(figsize=(7,4))
            try:
                for m in modes:
                    vals = dfm[dfm["mode"] == m][metric].values
                    sns.kdeplot(vals, label=m, fill=False)
                plt.title(metric)
                plt.legend()
                plt.tight_layout()
                out_png = os.path.join(args.outdir, f"{metric}_kde.png")
                plt.savefig(out_png, dpi=150)
                plt.close()
            except Exception:
                plt.close()
            # Box
            plt.figure(figsize=(6,4))
            try:
                sns.boxplot(data=dfm, x="mode", y=metric)
                plt.title(metric + " (box)")
                plt.tight_layout()
                out_png = os.path.join(args.outdir, f"{metric}_box.png")
                plt.savefig(out_png, dpi=150)
                plt.close()
            except Exception:
                plt.close()
            # Violin
            plt.figure(figsize=(6,4))
            try:
                sns.violinplot(data=dfm, x="mode", y=metric", cut=0)
                plt.title(metric + " (violin)")
                plt.tight_layout()
                out_png = os.path.join(args.outdir, f"{metric}_violin.png")
                plt.savefig(out_png, dpi=150)
                plt.close()
            except Exception:
                plt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()


