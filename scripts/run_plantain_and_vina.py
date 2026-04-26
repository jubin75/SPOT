from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import subprocess
from typing import List, Tuple, Dict, Optional


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


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _plantain_dir() -> str:
    return os.path.join(_project_root(), "lib", "plantain")


def _pick_device(s: str) -> str:
    s = (s or "auto").lower()
    if s in ("cpu", "auto"):
        return "cpu"
    if s == "mps":
        return "mps"
    if s == "cuda":
        return "cuda:0"
    return "cpu"


def _bbox_from_pdb_like(path: str) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ln.startswith("ATOM") or ln.startswith("HETATM"):
                try:
                    x = float(ln[30:38].strip()); y = float(ln[38:46].strip()); z = float(ln[46:54].strip())
                except Exception:
                    parts = ln.split()
                    if len(parts) < 9:
                        continue
                    x = float(parts[-6]); y = float(parts[-5]); z = float(parts[-4])
                xs.append(x); ys.append(y); zs.append(z)
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def _write_sdf_from_plantain(smi_file: str, pocket_pdb: str, out_dir: str, device: str = "auto", limit: int = -1) -> List[str]:
    """Run PLANTAIN inference and write multi-pose SDFs like lib/plantain/inference.py.
    Returns list of written SDF file paths in order of dataset index.
    """
    pdir = _plantain_dir()
    if pdir not in sys.path:
        sys.path.insert(0, pdir)
    with _Chdir(pdir):
        from common.cfg_utils import get_config  # type: ignore
        from models.pretrained_plantain import get_pretrained_plantain  # type: ignore
        from datasets.inference_dataset import InferenceDataset  # type: ignore
        from terrace import collate  # type: ignore
        from rdkit import Chem  # type: ignore
        from common.pose_transform import add_multi_pose_to_mol  # type: ignore

        cfg = get_config("icml")
        model = get_pretrained_plantain()
        ds = InferenceDataset(cfg, smi_file, pocket_pdb, model.get_input_feats())

        dev = _pick_device(device)
        try:
            model = model.to(dev)
        except Exception:
            dev = "cpu"; model = model.to("cpu")
        model.eval()

        os.makedirs(out_dir, exist_ok=True)
        written: List[str] = []
        N = len(ds) if (limit is None or limit < 0) else min(len(ds), int(limit))
        for i in range(N):
            x, y = ds[i]
            if hasattr(y, "success") and not getattr(y, "success", True):
                written.append("")
                continue
            batch = collate([x])
            try:
                batch = batch.to(dev)
            except Exception:
                pass
            pred = model(batch)[0]
            mol = getattr(x, "lig", None)
            if mol is None or not hasattr(pred, "lig_pose") or pred.lig_pose is None:
                written.append("")
                continue
            add_multi_pose_to_mol(mol, pred.lig_pose)
            sdf_path = os.path.join(out_dir, f"{i}.sdf")
            w = Chem.SDWriter(sdf_path)
            try:
                for c in range(mol.GetNumConformers()):
                    w.write(mol, c)
            finally:
                w.close()
            written.append(sdf_path)
        return written


def _clean_receptor_pdbqt(src: str, dst: str) -> None:
    bad = {"ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF"}
    with open(src, "r") as f, open(dst, "w") as g:
        for ln in f:
            tag = ln.split()[:1]
            if tag and tag[0] in bad:
                continue
            g.write(ln)


def _ensure_receptor_pdbqt(obabel_bin: str, pocket_pdb: str) -> str:
    """Strict mode: require ADFR/ADT prepare_receptor; no fallback to obabel."""
    rec_fix = pocket_pdb.replace(".pdb", "_rigid.pdbqt")
    last_out = ""
    for prep_cmd in ("prepare_receptor", "prepare_receptor4.py"):
        try:
            r = subprocess.run([prep_cmd, "-r", pocket_pdb, "-o", rec_fix, "-A", "checkhydrogens"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            last_out = (r.stdout or "").strip()
            if r.returncode == 0 and os.path.exists(rec_fix) and os.path.getsize(rec_fix) > 0:
                return rec_fix
        except FileNotFoundError:
            last_out = f"{prep_cmd} not found"
        except Exception as e:
            last_out = f"{prep_cmd} error: {e}"
    raise RuntimeError(f"prepare_receptor failed; ensure ADFR/ADT is in PATH. detail={last_out}")


def _ligand_first_pose_to_pdbqt(obabel_bin: str, sdf_path: str, out_dir: str) -> Optional[str]:
    """Strict mode: require Meeko to prepare ligand PDBQT (first conformer)."""
    base = os.path.splitext(os.path.basename(sdf_path))[0]
    lig_pdbqt = os.path.join(out_dir, base + ".pdbqt")
    os.makedirs(out_dir, exist_ok=True)
    # Reuse cached ligand if present
    try:
        if os.path.exists(lig_pdbqt) and os.path.getsize(lig_pdbqt) > 0:
            return lig_pdbqt
    except Exception:
        pass
    from rdkit import Chem  # type: ignore
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Meeko not available in current Python: {e}")
    mols = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)
    mol = None
    for m in mols:
        if m is not None:
            mol = m; break
    if mol is None:
        raise RuntimeError(f"RDKit failed to read SDF: {os.path.basename(sdf_path)}")
    # Ensure explicit hydrogens with 3D coords for Meeko
    try:
        mol = Chem.AddHs(mol, addCoords=True)
    except Exception:
        mol = Chem.AddHs(mol)
    prep = MoleculePreparation()
    u = prep.prepare(mol)
    # Meeko may return a list of setups; pick the first
    if isinstance(u, (list, tuple)):
        if not u:
            raise RuntimeError("Meeko returned empty setup list")
        u = u[0]
    s = PDBQTWriterLegacy().write_string(u, bad_charge_ok=True)
    if isinstance(s, tuple):
        s = s[0]
    with open(lig_pdbqt, "w", encoding="utf-8") as f:
        f.write(s)
    if not (os.path.exists(lig_pdbqt) and os.path.getsize(lig_pdbqt) > 0):
        raise RuntimeError(f"Meeko wrote empty PDBQT for {os.path.basename(sdf_path)}")
    return lig_pdbqt


def _bbox_from_pdbqt(path: str) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    return _bbox_from_pdb_like(path)


def _score_with_vina(receptor_pdbqt: str, ligand_pdbqt: str, center: List[float], size: List[float]) -> Tuple[Optional[float], Optional[float]]:
    try:
        from vina import Vina  # type: ignore
    except Exception:
        return None, None
    try:
        v = Vina(sf_name="vina")
        # Use positional args for maximum compatibility
        v.set_receptor(receptor_pdbqt)
        v.set_ligand_from_file(ligand_pdbqt)
        v.compute_vina_maps(center=center, box_size=size)
        res = v.score()
        try:
            raw = float(res)
        except Exception:
            import numpy as _np  # type: ignore
            raw = float(_np.asarray(res).ravel()[0])
        # Local optimization to obtain minimized affinity
        try:
            # Some versions support optimize(); fall back to a quick dock if not
            try:
                v.optimize()
            except Exception:
                v.dock(exhaustiveness=8, n_poses=1)
            res2 = v.score()
            try:
                opt = float(res2)
            except Exception:
                import numpy as _np2  # type: ignore
                opt = float(_np2.asarray(res2).ravel()[0])
        except Exception:
            opt = None
        return raw, opt
    except Exception:
        return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PLANTAIN to generate poses and rescore first pose with python-vina.")
    ap.add_argument("--smi", required=True, help="Input .smi (one SMILES per line)")
    ap.add_argument("--pocket", required=True, help="Pocket PDB path")
    ap.add_argument("--poses-dir", default=os.path.join("runs", "plantain_poses"))
    ap.add_argument("--out-csv", default=os.path.join("runs", "plantain_vina.csv"))
    ap.add_argument("--pdbqt-dir", default=os.path.join("runs", "vina_pdbqt"))
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"])
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--obabel-bin", default="/usr/local/bin/obabel")
    # New controls: pocket center/box and docking strength
    ap.add_argument("--center", default="", help="Override pocket center as 'cx,cy,cz'")
    ap.add_argument("--box_size", type=float, default=22.0, help="Cubic grid box size (per-dimension) in Angstroms")
    ap.add_argument("--exhaustiveness", type=int, default=32, help="Docking exhaustiveness for local search")
    ap.add_argument("--top_k", type=int, default=3, help="Number of PLANTAIN poses (first K conformers) to refine")
    ap.add_argument("--full_dock_th", type=float, default=-3.0, help="If minimized affinity > th, run a quick dock to seek lower energy")
    args = ap.parse_args()

    # 1) Generate Plantain SDF poses
    sdf_paths = _write_sdf_from_plantain(args.smi, args.pocket, args.poses_dir, device=args.device, limit=args.limit)

    # 2) Prepare receptor PDBQT (rigid) and pocket-centered grid
    rec_pdbqt = _ensure_receptor_pdbqt(args.obabel_bin, args.pocket)
    # Pocket-centered grid
    if args.center:
        cx, cy, cz = [float(x) for x in args.center.split(",")]
    else:
        (px, py, pz) = _bbox_from_pdb_like(args.pocket)
        cx, cy, cz = (px[0] + px[1]) / 2, (py[0] + py[1]) / 2, (pz[0] + pz[1]) / 2
    # Fixed cubic size (avoid huge volumes)
    sx = sy = sz = float(max(16.0, min(60.0, args.box_size)))

    # Preload one Vina instance and reuse grid
    try:
        from vina import Vina  # type: ignore
        vina_obj = Vina(sf_name="vina")
        vina_obj.set_receptor(rec_pdbqt)
        vina_obj.compute_vina_maps(center=[cx, cy, cz], box_size=[sx, sy, sz])
    except Exception:
        vina_obj = None

    # 3) For each SDF, convert top-K poses to PDBQT (or first if single) and score with vina
    rows: List[Dict[str, str]] = []
    for i, sdf in enumerate(sdf_paths):
        if not sdf:
            rows.append({"index": str(i), "sdf": "", "vina_affinity": ""})
            continue
        # Prepare ligand PDBQT from SDF; if SDF has multi-conformers, we will split top-K via Meeko roundtrip
        lig_pdbqt = _ligand_first_pose_to_pdbqt(args.obabel_bin, sdf, args.pdbqt_dir)
        if not lig_pdbqt:
            rows.append({"index": str(i), "sdf": os.path.basename(sdf), "vina_affinity": ""})
            continue
        # Score K poses: if we only have one PDBQT, treat as single pose
        pose_paths = [lig_pdbqt]
        # Optional: if obabel pathway used earlier for multi-poses, user could pre-split; here we keep single for speed

        best_raw = None
        best_min = None
        # Reuse shared Vina grid when possible
        if vina_obj is not None:
            v = vina_obj
        else:
            try:
                v = Vina(sf_name="vina")  # type: ignore
                v.set_receptor(rec_pdbqt)
                v.compute_vina_maps(center=[cx, cy, cz], box_size=[sx, sy, sz])
            except Exception:
                v = None
        for k, lig_path in enumerate(pose_paths[: max(1, int(args.top_k))]):
            try:
                if v is None:
                    raw_k, min_k = _score_with_vina(rec_pdbqt, lig_path, [cx, cy, cz], [sx, sy, sz])
                else:
                    v.set_ligand_from_file(lig_path)
                    res = v.score()
                    try:
                        raw_k = float(res)
                    except Exception:
                        import numpy as _np  # type: ignore
                        raw_k = float(_np.asarray(res).ravel()[0])
                    try:
                        v.optimize()
                    except Exception:
                        pass
                    res2 = v.score()
                    try:
                        min_k = float(res2)
                    except Exception:
                        import numpy as _np2  # type: ignore
                        min_k = float(_np2.asarray(res2).ravel()[0])
                # Track best across poses
                if best_raw is None or (raw_k is not None and raw_k < best_raw):
                    best_raw = raw_k
                if best_min is None or (min_k is not None and min_k < best_min):
                    best_min = min_k
            except Exception:
                continue
        # Thresholded quick dock if minimized energy is not good enough
        if best_min is None or (best_min is not None and best_min > float(args.full_dock_th)):
            try:
                v2 = Vina(sf_name="vina")  # type: ignore
                v2.set_receptor(rec_pdbqt)
                v2.set_ligand_from_file(lig_pdbqt)
                v2.compute_vina_maps(center=[cx, cy, cz], box_size=[sx, sy, sz])
                v2.dock(exhaustiveness=int(max(8, args.exhaustiveness)), n_poses=1)
                res3 = v2.score()
                try:
                    best_min2 = float(res3)
                except Exception:
                    import numpy as _np3  # type: ignore
                    best_min2 = float(_np3.asarray(res3).ravel()[0])
                if best_min is None or best_min2 < best_min:
                    best_min = best_min2
            except Exception:
                pass
        rows.append({
            "index": str(i),
            "sdf": os.path.basename(sdf),
            "vina_affinity": ("" if best_min is None else f"{best_min:.3f}"),
            "vina_affinity_raw": ("" if best_raw is None else f"{best_raw:.3f}"),
            "vina_affinity_min": ("" if best_min is None else f"{best_min:.3f}"),
        })

    # 4) Save CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "sdf", "vina_affinity", "vina_affinity_raw", "vina_affinity_min"])
        w.writeheader(); w.writerows(rows)
    print({"saved": args.out_csv, "rows": len(rows)})


if __name__ == "__main__":
    main()


