from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Tuple, Optional, Dict


def _parse_vec3(s: str) -> List[float]:
    return [float(x) for x in s.split(",")]


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


def _ensure_meeko_pdbqt_from_sdf(sdf_path: str, out_path: str) -> str:
    from rdkit import Chem  # type: ignore
    from meeko import MoleculePreparation, PDBQTWriterLegacy  # type: ignore
    mols = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)
    mol = None
    for m in mols:
        if m is not None:
            mol = m; break
    if mol is None:
        raise RuntimeError(f"RDKit failed to read SDF: {os.path.basename(sdf_path)}")
    try:
        mol = Chem.AddHs(mol, addCoords=True)
    except Exception:
        mol = Chem.AddHs(mol)
    prep = MoleculePreparation()
    u = prep.prepare(mol)
    if isinstance(u, (list, tuple)):
        if not u:
            raise RuntimeError("Meeko returned empty setup list")
        u = u[0]
    s = PDBQTWriterLegacy().write_string(u, bad_charge_ok=True)
    if isinstance(s, tuple):
        s = s[0]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(s)
    return out_path


def _vina_api_scores(rec_pdbqt: str, lig_pdbqt: str, center: List[float], size: List[float]) -> Tuple[Optional[float], Optional[float]]:
    try:
        from vina import Vina  # type: ignore
    except Exception:
        return None, None
    try:
        v = Vina(sf_name="vina")
        v.set_receptor(rec_pdbqt)
        v.set_ligand_from_file(lig_pdbqt)
        v.compute_vina_maps(center=center, box_size=size)
        res = v.score()
        try:
            raw = float(res)
        except Exception:
            import numpy as _np
            raw = float(_np.asarray(res).ravel()[0])
        try:
            v.optimize()
        except Exception:
            v.dock(exhaustiveness=16, n_poses=1)
        res2 = v.score()
        try:
            opt = float(res2)
        except Exception:
            import numpy as _np2
            opt = float(_np2.asarray(res2).ravel()[0])
        return raw, opt
    except Exception:
        return None, None


def _vina_cli_score(vina_bin: str, rec_pdbqt: str, lig_pdbqt: str, center: List[float], size: List[float]) -> Optional[float]:
    if not vina_bin:
        return None
    cmd = [vina_bin,
           "--receptor", rec_pdbqt, "--ligand", lig_pdbqt,
           "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
           "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
           "--score_only", "--cpu", "1"]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = r.stdout or ""
        if r.returncode != 0:
            return None
        for ln in out.splitlines():
            if ln.lower().startswith("affinity:"):
                try:
                    return float(ln.split()[1])
                except Exception:
                    continue
        return None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Vina grid coverage and cross-check API vs CLI; export optimized pose.")
    ap.add_argument("--receptor", required=True, help="Rigid receptor PDBQT (ADFR prepared)")
    ap.add_argument("--pocket", required=False, default="", help="Pocket PDB (to derive center/size if not provided)")
    ap.add_argument("--ligand_pdbqt", required=False, default="", help="Ligand PDBQT path")
    ap.add_argument("--sdf", required=False, default="", help="If provided, will prepare ligand PDBQT via Meeko")
    ap.add_argument("--center", required=False, default="", help="Center as cx,cy,cz; overrides pocket-based center if given")
    ap.add_argument("--size", required=False, default="", help="Size as sx,sy,sz; overrides pocket-based size if given")
    ap.add_argument("--vina_bin", required=False, default="", help="CLI vina path for score_only cross-check")
    ap.add_argument("--out_dir", required=False, default=os.path.join("runs", "vina_debug"))
    args = ap.parse_args()

    # 1) Resolve ligand PDBQT
    lig = args.ligand_pdbqt.strip()
    if not lig:
        if not args.sdf:
            raise SystemExit("Provide either --ligand_pdbqt or --sdf")
        base = os.path.splitext(os.path.basename(args.sdf))[0]
        lig = os.path.join(args.out_dir, f"{base}_prepared.pdbqt")
        os.makedirs(args.out_dir, exist_ok=True)
        lig = _ensure_meeko_pdbqt_from_sdf(args.sdf, lig)

    # 2) Grid center/size
    if args.center:
        center = _parse_vec3(args.center)
    elif args.pocket:
        (px, py, pz) = _bbox_from_pdb_like(args.pocket)
        center = [(px[0]+px[1])/2, (py[0]+py[1])/2, (pz[0]+pz[1])/2]
    else:
        raise SystemExit("Need --center or --pocket to compute center")

    if args.size:
        size = _parse_vec3(args.size)
    elif args.pocket:
        (px, py, pz) = _bbox_from_pdb_like(args.pocket)
        BASE_CAP, MAX_CAP = 22.0, 30.0
        sx = min(MAX_CAP, max(BASE_CAP, (px[1]-px[0]) + 8.0))
        sy = min(MAX_CAP, max(BASE_CAP, (py[1]-py[0]) + 8.0))
        sz = min(MAX_CAP, max(BASE_CAP, (pz[1]-pz[0]) + 8.0))
        size = [sx, sy, sz]
    else:
        raise SystemExit("Need --size or --pocket to compute size")

    # 3) Coverage check
    (lx, ly, lz) = _bbox_from_pdb_like(lig)
    half = [size[0]/2.0, size[1]/2.0, size[2]/2.0]
    rmax = [max(abs(lx[0]-center[0]), abs(lx[1]-center[0])),
            max(abs(ly[0]-center[1]), abs(ly[1]-center[1])),
            max(abs(lz[0]-center[2]), abs(lz[1]-center[2]))]
    inside = [rmax[i] <= half[i] for i in range(3)]
    print({"center": center, "size": size, "half": half, "rmax": rmax, "inside": inside})

    # 4) API score (raw and minimized)
    api_raw, api_min = _vina_api_scores(args.receptor, lig, center, size)
    print({"vina_api_raw": api_raw, "vina_api_min": api_min})

    # 5) CLI cross-check
    cli = _vina_cli_score(args.vina_bin, args.receptor, lig, center, size) if args.vina_bin else None
    print({"vina_cli_score_only": cli})

    # 6) Export minimized pose for PyMOL
    try:
        from vina import Vina  # type: ignore
        v = Vina(sf_name="vina")
        v.set_receptor(args.receptor)
        v.set_ligand_from_file(lig)
        v.compute_vina_maps(center=center, box_size=size)
        try:
            v.optimize()
        except Exception:
            v.dock(exhaustiveness=16, n_poses=1)
        base = os.path.splitext(os.path.basename(lig))[0]
        out_pose = os.path.join(args.out_dir, f"{base}_opt.pdbqt")
        os.makedirs(args.out_dir, exist_ok=True)
        v.write_poses(out_pose, n_poses=1)
        print({"saved_pose": out_pose})
    except Exception:
        pass


if __name__ == "__main__":
    main()


