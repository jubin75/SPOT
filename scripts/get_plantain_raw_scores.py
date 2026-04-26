from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Iterable, List, Optional, Tuple


class _Chdir:
    """Temporarily change working directory."""
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
    dev = "cpu"
    d = (device_arg or "auto").lower()
    if d == "cpu":
        return "cpu"
    if d == "mps":
        return "mps"
    if d == "cuda":
        return "cuda:0"
    # auto
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return dev


def _load_model_and_dataset(
    smi_file: str,
    pocket_pdb: str,
    device: str,
):
    """Load PLANTAIN model and build inference dataset.

    Returns (model, dataset, collate_fn) or (None, None, None) on failure.
    """
    plantain_dir = _get_plantain_dir()
    try:
        if plantain_dir not in sys.path:
            sys.path.insert(0, plantain_dir)
        with _Chdir(plantain_dir):
            from common.cfg_utils import get_config  # type: ignore
            from models.pretrained_plantain import get_pretrained_plantain  # type: ignore
            from datasets.inference_dataset import InferenceDataset  # type: ignore
            from terrace import collate  # type: ignore

            cfg = get_config("icml")
            model = get_pretrained_plantain()
            try:
                import torch  # type: ignore
                model = model.to(device)
            except Exception:
                pass
            model.eval()

            dataset = InferenceDataset(cfg, smi_file, pocket_pdb, model.get_input_feats())
            return model, dataset, collate
    except Exception:
        return None, None, None
    return None, None, None


def _read_smiles_list(smi_path: str) -> List[str]:
    with open(smi_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    # accept whitespace/comments; keep non-empty, non-comment
    smiles: List[str] = [ln.split()[0] for ln in lines if ln and not ln.startswith("#")]
    return smiles


def compute_raw_scores(
    smi_file: str,
    pocket_pdb: str,
    device: str = "auto",
    limit: int = -1,
    sdf_out_dir: Optional[str] = None,
    verbose: bool = True,
) -> List[Tuple[str, Optional[float], Optional[List[float]]]]:
    """Compute raw PLANTAIN pose scores per ligand.

    Returns list of (smiles, min_score, all_scores).
    """
    dev = _pick_device(device)
    model, dataset, collate_fn = _load_model_and_dataset(smi_file, pocket_pdb, dev)
    if model is None or dataset is None or collate_fn is None:
        return []
    if verbose:
        print({"device": dev, "dataset_size": len(dataset)})

    # Disable torch.compile to improve compatibility on CPU/remote envs
    try:
        if hasattr(model, "cfg") and hasattr(model.cfg, "platform"):
            try:
                model.cfg.platform.compile = False  # type: ignore[attr-defined]
            except Exception:
                try:
                    model.cfg.platform["compile"] = False  # type: ignore[index]
                except Exception:
                    pass
    except Exception:
        pass

    smiles_list = _read_smiles_list(smi_file)
    results: List[Tuple[str, Optional[float], Optional[List[float]]]] = []

    n = len(dataset)
    if limit is not None and limit > 0:
        n = min(n, int(limit))

    # Lazy import torch to avoid hard dependency if environment lacks it
    try:
        import torch  # type: ignore
    except Exception:
        torch = None  # type: ignore

    # Optional SDF export setup
    if sdf_out_dir:
        try:
            os.makedirs(sdf_out_dir, exist_ok=True)
        except Exception:
            sdf_out_dir = None
    plantain_dir = _get_plantain_dir()

    for i in range(n):
        try:
            # Ensure dataset item construction (with transforms) runs under PLANTAIN cwd
            with _Chdir(plantain_dir):
                x, y = dataset[i]
            # y.success can be missing; guard with getattr
            if hasattr(y, "success") and not getattr(y, "success", True):
                if verbose:
                    print({"index": i, "status": "embed_failed"})
                results.append((smiles_list[i] if i < len(smiles_list) else "", None, None))
                continue
            batch = collate_fn([x])
            try:
                if torch is not None:
                    batch = batch.to(dev)
            except Exception:
                pass
            # Run inference inside plantain repo cwd to satisfy any relative paths
            with _Chdir(plantain_dir):
                pred = model(batch)[0]

            scores_tensor = getattr(pred, "score", None)

            # Fallback: explicitly run infer_bfgs if pred.score is missing
            if scores_tensor is None:
                try:
                    with _Chdir(plantain_dir):
                        hid_feat = model.get_hidden_feat(batch)
                        lig_pose, score_tensor = model.infer_bfgs(batch, hid_feat)
                    # Align with pred-like structure for SDF export below
                    pred = type("_P", (), {"lig_pose": lig_pose, "score": score_tensor})()
                    scores_tensor = score_tensor
                except Exception:
                    if verbose:
                        print({"index": i, "status": "infer_bfgs_failed"})
                    results.append((smiles_list[i] if i < len(smiles_list) else "", None, None))
                    continue
            try:
                # Ensure on CPU and python floats
                scores_list: List[float] = [
                    float(s.item()) if hasattr(s, "item") else float(s)
                    for s in scores_tensor.detach().cpu().flatten()
                ]
            except Exception:
                # As a fallback, try naive conversion
                scores_list = [float(v) for v in list(scores_tensor)]  # type: ignore

            min_score: Optional[float] = min(scores_list) if scores_list else None
            if verbose and i < 3:
                print({"index": i, "min_score": min_score})

            # Optional SDF export with multi-poses (like lib/plantain/inference.py)
            if sdf_out_dir:
                try:
                    if plantain_dir not in sys.path:
                        sys.path.insert(0, plantain_dir)
                    from rdkit import Chem  # type: ignore
                    from common.pose_transform import add_multi_pose_to_mol  # type: ignore
                    mol = getattr(x, "lig", None)
                    if mol is not None and hasattr(pred, "lig_pose") and pred.lig_pose is not None:
                        add_multi_pose_to_mol(mol, pred.lig_pose)
                        sdf_path = os.path.join(sdf_out_dir, f"{i}.sdf")
                        # Ensure write while inside plantain cwd as well
                        with _Chdir(plantain_dir):
                            writer = Chem.SDWriter(sdf_path)
                            try:
                                for c in range(mol.GetNumConformers()):
                                    writer.write(mol, c)
                            finally:
                                writer.close()
                        if verbose and i < 3:
                            print({"index": i, "sdf": sdf_path})
                except Exception:
                    # Ignore SDF export failures but keep scores
                    pass

            results.append((smiles_list[i] if i < len(smiles_list) else "", min_score, scores_list))
        except Exception:
            results.append((smiles_list[i] if i < len(smiles_list) else "", None, None))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute raw PLANTAIN minimal scores from a .smi against a pocket PDB")
    parser.add_argument("--smi", required=True, type=str, help="Input .smi file (one SMILES per line)")
    parser.add_argument("--pocket", required=True, type=str, help="Pocket PDB file path")
    parser.add_argument("--out", required=False, type=str, default=os.path.join(_get_project_root(), "runs", "plantain_raw_scores.csv"), help="Output CSV path")
    parser.add_argument("--device", required=False, type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device selection")
    parser.add_argument("--limit", required=False, type=int, default=-1, help="Optional limit on number of ligands to score")
    parser.add_argument("--include_all_scores", action="store_true", help="Include all pose scores as a semicolon-separated string column")
    parser.add_argument("--sdf_out", required=False, type=str, default="", help="Optional folder to export multi-pose SDFs per ligand")
    parser.add_argument("--quiet", action="store_true", help="Silence progress logs")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = compute_raw_scores(
        args.smi,
        args.pocket,
        device=args.device,
        limit=int(args.limit),
        sdf_out_dir=(args.sdf_out or None),
        verbose=(not args.quiet),
    )

    fieldnames = ["smiles", "plantain_min"]
    if args.include_all_scores:
        fieldnames.append("pose_scores")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for smi, min_score, all_scores in rows:
            rec = {
                "smiles": smi,
                "plantain_min": ("" if min_score is None else f"{min_score:.6f}"),
            }
            if args.include_all_scores:
                rec["pose_scores"] = (
                    "" if not all_scores else ";".join(f"{v:.6f}" for v in all_scores)
                )
            writer.writerow(rec)

    print({"saved": args.out, "rows": len(rows)})


if __name__ == "__main__":
    main()


