from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


def _log(msg: str) -> None:
    print(msg, flush=True)


def _list_pockets(test_dir: str) -> Dict[str, str]:
    """Return mapping pdb_id -> pocket_pdb path."""
    out: Dict[str, str] = {}
    for p in glob.glob(os.path.join(test_dir, "*", "*_pocket.pdb")):
        base = os.path.basename(p)
        m = re.match(r"([A-Za-z0-9]{4})_pocket\.pdb$", base)
        if m:
            out[m.group(1).lower()] = p
    return out


def _infer_pdb_id_from_filename(fname: str, candidates: Dict[str, str]) -> Optional[str]:
    name = os.path.basename(fname).lower()
    hits = [pid for pid in candidates.keys() if pid in name]
    if len(hits) == 1:
        return hits[0]
    return None


def _ensure_receptor_pdbqt(pocket_pdb: str, obabel_bin: str = "obabel") -> Optional[str]:
    pdbqt = os.path.splitext(pocket_pdb)[0] + ".pdbqt"
    if os.path.exists(pdbqt) and os.path.getsize(pdbqt) > 0:
        return pdbqt
    cmd = [obabel_bin, "-ipdb", pocket_pdb, "-opdbqt", "-O", pdbqt]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode == 0 and os.path.exists(pdbqt) and os.path.getsize(pdbqt) > 0:
            return pdbqt
        _log(f"[WARN] obabel receptor failed ({pocket_pdb}). Output: {r.stdout.strip()}")
        return None
    except FileNotFoundError:
        _log(f"[ERROR] obabel not found ({obabel_bin}) when converting receptor")
        return None


def _convert_ligand_to_pdbqt_first_pose(sdf_path: str, out_dir: str, obabel_bin: str = "obabel") -> Optional[str]:
    base = os.path.splitext(os.path.basename(sdf_path))[0]
    pdbqt = os.path.join(out_dir, base + ".pdbqt")
    os.makedirs(out_dir, exist_ok=True)
    # Robust approach: write first pose to a temporary SDF using RDKit, then convert with explicit formats
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import SDWriter  # type: ignore
        suppl = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
        mol = None
        for m in suppl:
            if m is not None:
                mol = m
                break
        if mol is None:
            _log(f"[WARN] RDKit failed to read any molecule from {os.path.basename(sdf_path)}")
            return None
        # Keep only the first conformer
        if mol.GetNumConformers() > 1:
            conf0 = mol.GetConformer(0)
            nm = Chem.Mol(mol)
            nm.RemoveAllConformers()
            nm.AddConformer(conf0, assignId=True)
            mol = nm
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_sdf = os.path.join(td, "first_pose.sdf")
            w = SDWriter(tmp_sdf)
            try:
                w.write(mol)
            finally:
                w.close()
            # Explicitly specify input/output formats and add hydrogens
            cmd = [obabel_bin, "-isdf", tmp_sdf, "-opdbqt", "-O", pdbqt, "-h"]
            try:
                r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if r.returncode == 0 and os.path.exists(pdbqt) and os.path.getsize(pdbqt) > 0:
                    return pdbqt
                _log(f"[WARN] obabel ligand failed ({os.path.basename(sdf_path)}). Output: {r.stdout.strip()}")
            except FileNotFoundError:
                _log(f"[ERROR] obabel not found ({obabel_bin}) when converting ligand")
                return None
    except Exception as e:
        _log(f"[WARN] RDKit pre-processing failed for {os.path.basename(sdf_path)}: {e}")
        # Final fallback: direct obabel with explicit format and first/last selection
        cmd = [obabel_bin, "-isdf", sdf_path, "-opdbqt", "-O", pdbqt, "-f", "1", "-l", "1", "-h"]
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if r.returncode == 0 and os.path.exists(pdbqt) and os.path.getsize(pdbqt) > 0:
                return pdbqt
            _log(f"[WARN] obabel ligand failed ({os.path.basename(sdf_path)}). Output: {r.stdout.strip()}")
        except FileNotFoundError:
            _log(f"[ERROR] obabel not found ({obabel_bin}) when converting ligand")
            return None
    return None


def _compute_box_from_pocket(pocket_pdb: str, margin: float = 8.0, min_size: float = 16.0) -> Tuple[List[float], List[float]]:
    """Compute docking box from PDB by parsing ATOM/HETATM XYZ columns (no Biopython required).

    PDB fixed columns: X[30:38], Y[38:46], Z[46:54]
    """
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    try:
        with open(pocket_pdb, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
                    continue
                try:
                    x = float(ln[30:38].strip())
                    y = float(ln[38:46].strip())
                    z = float(ln[46:54].strip())
                except Exception:
                    # Fallback to whitespace split if fixed-width parse fails
                    parts = ln.split()
                    if len(parts) < 9:
                        continue
                    # Typical PDB fields end with x y z occupancy temp
                    try:
                        x = float(parts[-6]); y = float(parts[-5]); z = float(parts[-4])
                    except Exception:
                        continue
                xs.append(x); ys.append(y); zs.append(z)
    except Exception:
        pass
    if not xs:
        return [0.0, 0.0, 0.0], [min_size, min_size, min_size]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    cz = 0.5 * (min_z + max_z)
    sx = max(min_size, (max_x - min_x) + margin)
    sy = max(min_size, (max_y - min_y) + margin)
    sz = max(min_size, (max_z - min_z) + margin)
    return [cx, cy, cz], [sx, sy, sz]


def _vina_score_py(receptor_pdbqt: str, ligand_pdbqt: str, center: List[float], size: List[float]) -> Optional[float]:
    try:
        from vina import Vina  # type: ignore
    except Exception as e:
        _log(f"[WARN] vina Python import failed: {e}")
        return None
    try:
        v = Vina(sf_name='vina', verbosity=0)
        # Be robust to API differences across vina versions
        set_rec_ok = False
        try:
            v.set_receptor(receptor_pdbqt=receptor_pdbqt)
            set_rec_ok = True
        except TypeError:
            try:
                v.set_receptor(rigid_pdbqt=receptor_pdbqt)
                set_rec_ok = True
            except Exception:
                try:
                    v.set_receptor(pdbqt_filename=receptor_pdbqt)
                    set_rec_ok = True
                except Exception:
                    set_rec_ok = False
        if not set_rec_ok:
            _log(f"[WARN] vina Python set_receptor failed for {os.path.basename(receptor_pdbqt)}")
            return None

        set_lig_ok = False
        try:
            v.set_ligand_from_file(ligand_pdbqt=ligand_pdbqt)
            set_lig_ok = True
        except TypeError:
            try:
                v.set_ligand_from_file(pdbqt_filename=ligand_pdbqt)
                set_lig_ok = True
            except Exception:
                try:
                    v.set_ligand_from_file(ligand_pdbqt)
                    set_lig_ok = True
                except Exception:
                    set_lig_ok = False
        if not set_lig_ok:
            _log(f"[WARN] vina Python set_ligand_from_file failed for {os.path.basename(ligand_pdbqt)}")
            return None

        v.compute_vina_maps(center=center, box_size=size)
        s = v.score(exhaustiveness=8)
        return float(s)
    except Exception as e:
        _log(f"[WARN] vina Python scoring failed for {os.path.basename(ligand_pdbqt)}: {e}")
        return None


def _vina_score_cli(receptor_pdbqt: str, ligand_pdbqt: str, center: List[float], size: List[float], vina_bin: str = "vina") -> Optional[float]:
    cmd = [
        vina_bin,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--center_x", str(float(center[0])),
        "--center_y", str(float(center[1])),
        "--center_z", str(float(center[2])),
        "--size_x", str(float(size[0])),
        "--size_y", str(float(size[1])),
        "--size_z", str(float(size[2])),
        "--score_only",
        "--cpu", "1",
    ]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = r.stdout or ""
        if r.returncode != 0:
            _log(f"[WARN] vina CLI failed (rc={r.returncode}) for {os.path.basename(ligand_pdbqt)}: {out.strip()[:240]}")
            return None
        # Parse line like: "Affinity: -7.8 (kcal/mol)"
        for line in out.splitlines():
            line = line.strip()
            if line.lower().startswith("affinity:"):
                try:
                    val = float(line.split()[1])
                    return val
                except Exception:
                    continue
        _log(f"[WARN] vina CLI output parse failed for {os.path.basename(ligand_pdbqt)}")
        return None


def _vina_score_via_interpreter(
    vina_python: str,
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    center: List[float],
    size: List[float],
) -> Optional[float]:
    """Run python-vina in a separate interpreter that has vina installed."""
    try:
        cmd = [
            vina_python,
            os.path.abspath(__file__),
            "--worker-vina-score",
            "--receptor", receptor_pdbqt,
            "--ligand", ligand_pdbqt,
            "--center_x", str(float(center[0])),
            "--center_y", str(float(center[1])),
            "--center_z", str(float(center[2])),
            "--size_x", str(float(size[0])),
            "--size_y", str(float(size[1])),
            "--size_z", str(float(size[2]))
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = (r.stdout or "").strip()
        if r.returncode != 0:
            _log(f"[WARN] vina worker failed rc={r.returncode}: {out[:240]}")
            return None
        try:
            return float(out.splitlines()[-1].strip()) if out else None
        except Exception:
            _log(f"[WARN] vina worker parse failed: {out[:240]}")
            return None
    except Exception as e:
        _log(f"[WARN] vina worker spawn failed: {e}")
        return None


def _worker_vina_score_entry(args) -> None:
    """Worker mode: run inside an interpreter that has python-vina installed and print score."""
    try:
        from vina import Vina  # type: ignore
    except Exception as e:
        print(f"vina_import_error: {e}")
        raise SystemExit(1)
    try:
        v = Vina(sf_name='vina', verbosity=0)
        # receptor
        set_rec_ok = False
        for kw in ("receptor_pdbqt", "rigid_pdbqt", "pdbqt_filename"):
            try:
                getattr(v, "set_receptor")(**{kw: args.receptor})
                set_rec_ok = True
                break
            except Exception:
                continue
        if not set_rec_ok:
            print("set_receptor_error")
            raise SystemExit(2)
        # ligand
        set_lig_ok = False
        for kw in ("ligand_pdbqt", "pdbqt_filename"):
            try:
                v.set_ligand_from_file(**{kw: args.ligand})
                set_lig_ok = True
                break
            except Exception:
                continue
        if not set_lig_ok:
            print("set_ligand_error")
            raise SystemExit(3)
        center = [args.center_x, args.center_y, args.center_z]
        size = [args.size_x, args.size_y, args.size_z]
        v.compute_vina_maps(center=center, box_size=size)
        s = v.score(exhaustiveness=8)
        print(float(s))
        raise SystemExit(0)
    except SystemExit as se:
        raise se
    except Exception as e:
        print(f"vina_worker_error: {e}")
        raise SystemExit(4)
    except FileNotFoundError:
        _log(f"[ERROR] vina CLI not found ({vina_bin})")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Rescore Plantain poses with AutoDock Vina (score-only)")
    ap.add_argument("--poses-dir", type=str, default=os.path.join("run", "plantain_poses"))
    ap.add_argument("--test-dir", type=str, default="test")
    ap.add_argument("--out", type=str, default=os.path.join("run", "vina_rescored.csv"))
    ap.add_argument("--pdbqt-dir", type=str, default=os.path.join("run", "vina_pdbqt"), help="Where to write ligand PDBQT files")
    ap.add_argument("--vina-bin", type=str, default="vina", help="Path to vina binary (CLI)")
    ap.add_argument("--obabel-bin", type=str, default="obabel", help="Path to obabel binary")
    ap.add_argument("--vina-python", type=str, default="", help="Path to python interpreter that has python-vina installed")
    ap.add_argument("--vina-site", type=str, default="", help="Site-packages directory containing python-vina (prepend to sys.path). Use pathsep to pass multiple.")
    # Worker mode (internal use)
    ap.add_argument("--worker-vina-score", action="store_true")
    ap.add_argument("--receptor", type=str, default="")
    ap.add_argument("--ligand", type=str, default="")
    ap.add_argument("--center_x", type=float, default=0.0)
    ap.add_argument("--center_y", type=float, default=0.0)
    ap.add_argument("--center_z", type=float, default=0.0)
    ap.add_argument("--size_x", type=float, default=0.0)
    ap.add_argument("--size_y", type=float, default=0.0)
    ap.add_argument("--size_z", type=float, default=0.0)
    args = ap.parse_args()

    # Worker short-circuit
    if args.worker_vina_score:
        _worker_vina_score_entry(args)
        return

    # Optionally prepend site-packages for python-vina to current sys.path
    if args.vina_site:
        for p in args.vina_site.split(os.pathsep):
            p = p.strip()
            if p and os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        _log(f"[INFO] sys.path updated for vina_site: {args.vina_site}")

    pockets = _list_pockets(args.test_dir)
    if not pockets:
        _log(f"[ERROR] No pocket files found under {args.test_dir}/*/*_pocket.pdb")
        return
    _log(f"[INFO] Found pockets: {sorted(pockets.keys())}")

    sdf_files = sorted(glob.glob(os.path.join(args.poses_dir, "*.sdf")))
    if not sdf_files:
        _log(f"[ERROR] No SDF files found under {args.poses_dir}")
        return
    _log(f"[INFO] Found {len(sdf_files)} SDF files")

    rows: List[Dict[str, str]] = []

    # Cache receptor pdbqt per pdb_id
    rec_cache: Dict[str, Tuple[str, List[float], List[float]]] = {}

    for sdf in sdf_files:
        # Infer pdb_id
        pdb_id = _infer_pdb_id_from_filename(sdf, pockets)
        if pdb_id is None:
            if len(pockets) == 1:
                pdb_id = next(iter(pockets.keys()))
            else:
                _log(f"[WARN] Skip {os.path.basename(sdf)}: cannot infer pdb_id from filename")
                continue

        pocket_pdb = pockets[pdb_id]
        # Prepare receptor and grid
        if pdb_id not in rec_cache:
            rec_pdbqt = _ensure_receptor_pdbqt(pocket_pdb, obabel_bin=args.obabel_bin)
            if rec_pdbqt is None:
                _log(f"[WARN] Skip {os.path.basename(sdf)}: receptor pdbqt conversion failed")
                continue
            center, size = _compute_box_from_pocket(pocket_pdb)
            rec_cache[pdb_id] = (rec_pdbqt, center, size)
            _log(f"[INFO] Grid for {pdb_id}: center={center}, size={size}")
        rec_pdbqt, center, size = rec_cache[pdb_id]

        # Convert ligand (first pose)
        lig_pdbqt = _convert_ligand_to_pdbqt_first_pose(sdf, args.pdbqt_dir, obabel_bin=args.obabel_bin)
        if lig_pdbqt is None:
            _log(f"[WARN] Skip {os.path.basename(sdf)}: ligand pdbqt conversion failed")
            continue

        # Score with vina
        score = None
        # Prefer python-vina in a specified interpreter if provided
        if args.vina_python:
            score = _vina_score_via_interpreter(args.vina_python, rec_pdbqt, lig_pdbqt, center, size)
        if score is None:
            # Try python-vina in current interpreter
            score = _vina_score_py(rec_pdbqt, lig_pdbqt, center, size)
        if score is None:
            # Fallback to CLI
            score = _vina_score_cli(rec_pdbqt, lig_pdbqt, center, size, vina_bin=args.vina_bin)
        rows.append({
            "ligand_sdf": os.path.basename(sdf),
            "pdb_id": pdb_id,
            "vina_score": ("" if score is None else f"{float(score):.3f}"),
        })
        _log(f"[INFO] {os.path.basename(sdf)} -> vina_score={rows[-1]['vina_score']}")

    # Save CSV
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ligand_sdf", "pdb_id", "vina_score"])
        w.writeheader(); w.writerows(rows)
    _log({"saved": args.out, "rows": len(rows)})


if __name__ == "__main__":
    main()


