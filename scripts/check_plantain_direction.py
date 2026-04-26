from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Tuple, Optional


class _Chdir:
    def __init__(self, new_dir: str) -> None:
        self._new_dir = new_dir
        self._old_dir = os.getcwd()

    def __enter__(self):
        try:
            os.chdir(self._new_dir)
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            os.chdir(self._old_dir)
        except Exception:
            pass


def _get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _get_plantain_dir() -> str:
    return os.path.join(_get_project_root(), "lib", "plantain")


def _pick_device(device_arg: str = "auto") -> str:
    d = (device_arg or "auto").lower()
    # For robustness, default to CPU unless user explicitly requests otherwise
    if d in ("cpu", "auto"):
        return "cpu"
    if d == "mps":
        return "mps"
    if d == "cuda":
        return "cuda:0"
    return "cpu"


def _count_heavy_atoms_from_smiles(smi: str) -> Optional[int]:
    try:
        from rdkit import Chem  # type: ignore
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return int(mol.GetNumHeavyAtoms())
    except Exception:
        return None


def compute_min_scores_via_oracle(smi_file: str, pocket_pdb: str, device: str = "auto") -> List[Tuple[str, Optional[float], Optional[int], Optional[float]]]:
    """Use the same helper as training/online inference to compute minimal PLANTAIN score per SMILES.

    Returns (smiles, plantain_min, n_heavy_atoms, calibrated_score) where calibrated_score = score / sqrt(n_heavy_atoms).
    """
    from LeadGFlowNet import oracle as O  # local project import

    dev = _pick_device(device)
    with open(smi_file, "r", encoding="utf-8") as f:
        smiles = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    rows: List[Tuple[str, Optional[float], Optional[int], Optional[float]]] = []
    plantain_dir = _get_plantain_dir()
    for s in smiles:
        try:
            # Ensure PLANTAIN is executed under its repo cwd for relative paths/configs
            with _Chdir(plantain_dir):
                val = O._plantain_min_score_for_smiles(s, pocket_pdb, device=dev)
        except Exception:
            val = None
        nh = _count_heavy_atoms_from_smiles(s)
        if val is None or nh is None or nh <= 0:
            cal = None
        else:
            import math as _m
            cal = float(val) / (_m.sqrt(float(max(1, nh))))
        rows.append((s, None if val is None else float(val), nh, cal))
    return rows


def export_sdf_for_indices(
    smi_file: str,
    pocket_pdb: str,
    indices: List[int],
    out_dir: str,
    device: str = "auto",
) -> None:
    """Replicate lib/plantain/inference.py behavior to export multi-pose SDFs for selected indices."""
    plantain_dir = _get_plantain_dir()
    # Resolve PLANTAIN model/dataset under its cwd
    if plantain_dir not in sys.path:
        sys.path.insert(0, plantain_dir)
    with _Chdir(plantain_dir):
        from common.cfg_utils import get_config  # type: ignore
        from models.pretrained_plantain import get_pretrained_plantain  # type: ignore
        from datasets.inference_dataset import InferenceDataset  # type: ignore
        from terrace import collate  # type: ignore
        from rdkit import Chem  # type: ignore
        from common.pose_transform import add_multi_pose_to_mol  # type: ignore

        cfg = get_config("icml")
        model = get_pretrained_plantain()
        try:
            import torch  # type: ignore
            model = model.to(_pick_device(device))
            # disable compile for compatibility
            try:
                model.cfg.platform["compile"] = False  # type: ignore[index]
            except Exception:
                pass
        except Exception:
            pass
        model.eval()

        dataset = InferenceDataset(cfg, smi_file, pocket_pdb, model.get_input_feats())
        os.makedirs(out_dir, exist_ok=True)

        for i in indices:
            if i < 0 or i >= len(dataset):
                continue
            try:
                x, y = dataset[i]
                if hasattr(y, "success") and not getattr(y, "success", True):
                    continue
                batch = collate([x])
                try:
                    batch = batch.to(_pick_device(device))
                except Exception:
                    pass
                pred = model(batch)[0]
                mol = getattr(x, "lig", None)
                if mol is None or not hasattr(pred, "lig_pose") or pred.lig_pose is None:
                    continue
                add_multi_pose_to_mol(mol, pred.lig_pose)
                sdf_path = os.path.join(out_dir, f"{i}.sdf")
                w = Chem.SDWriter(sdf_path)
                try:
                    for c in range(mol.GetNumConformers()):
                        w.write(mol, c)
                finally:
                    w.close()
            except Exception:
                # skip failures
                pass


def main() -> None:
    p = argparse.ArgumentParser(description="Check PLANTAIN score direction and optionally export SDFs for top-K under both conventions.")
    p.add_argument("--smi", required=True, type=str)
    p.add_argument("--pocket", required=True, type=str)
    p.add_argument("--out_csv", required=False, type=str, default=os.path.join(_get_project_root(), "runs", "plantain_direction_check.csv"))
    p.add_argument("--device", required=False, type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    p.add_argument("--k", required=False, type=int, default=20)
    p.add_argument("--export_sdf", action="store_true")
    p.add_argument("--out_dir", required=False, type=str, default=os.path.join(_get_project_root(), "runs", "plantain_direction_sdf"))
    args = p.parse_args()

    rows = compute_min_scores_via_oracle(args.smi, args.pocket, device=args.device)
    # Sort by calibrated score descending (higher is better after calibration)
    dsc_cal = sorted(
        enumerate(rows),
        key=lambda t: (-float("inf") if t[1][3] is None else -t[1][3])
    )

    # Write combined CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "smiles","plantain_min","n_heavy_atoms","new_score","rank_desc"
        ])
        w.writeheader()
        # rank only by calibrated score (new_score) descending
        rank_map = {idx: r for r, (idx, _) in enumerate(dsc_cal, start=1)}
        for idx, (smi, val, nh, cal) in enumerate(rows):
            w.writerow({
                "smiles": smi,
                "plantain_min": ("" if val is None else f"{val:.6f}"),
                "n_heavy_atoms": ("" if nh is None else int(nh)),
                "new_score": ("" if cal is None else f"{cal:.6f}"),
                "rank_desc": rank_map.get(idx, ""),
            })

    print({"saved_csv": args.out_csv, "n": len(rows)})

    if args.export_sdf:
        # Export SDFs for top-K under the single convention: new_score descending
        k = max(1, int(args.k))
        dsc_cal_top_idx = [idx for idx, _ in [t for t in dsc_cal[:k]]]
        out_score_desc = os.path.join(args.out_dir, "score_desc")
        # Run export inside PLANTAIN cwd to mirror official CLI behavior
        with _Chdir(_get_plantain_dir()):
            export_sdf_for_indices(args.smi, args.pocket, dsc_cal_top_idx, out_score_desc, device=args.device)
        print({"sdf_score_desc": out_score_desc, "k": k})


if __name__ == "__main__":
    main()


