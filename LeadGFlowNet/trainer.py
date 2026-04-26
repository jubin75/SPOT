from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional
import os
import sys

import torch
from torch import nn

from LeadGFlowNet.conditional_policy import ConditionalSynthPolicy
from LeadGFlowNet.qsar import QSARPredictor
from LeadGFlowNet import oracle


@dataclass
class Trajectory:
    states: List
    actions_block: List[int]
    actions_rxn: List[int]
    terminal_smiles: str
    log_pf: torch.Tensor  # sum log forward policy
    log_pb: torch.Tensor  # sum log backward policy (simple uniform placeholder)


class LeadGFlowNetTrainer:
    def __init__(
        self,
        policy: ConditionalSynthPolicy,
        log_z: nn.Parameter,
        reward_fn: Callable[[str, str], float],
        protein_seq: str,
        device: torch.device,
    ) -> None:
        self.policy = policy
        self.log_z = log_z
        self.reward_fn = reward_fn
        self.protein_seq = protein_seq
        self.device = device

    def trajectory_balance_loss(self, traj: Trajectory, reward: float) -> torch.Tensor:
        # TB: (log Z + sum log P_F - log R - sum log P_B)^2
        log_r = torch.log(torch.tensor(max(reward, 1e-12), device=self.device, dtype=torch.float32))
        return (self.log_z + traj.log_pf - log_r - traj.log_pb) ** 2

    def train_step(self, batch_trajs: List[Trajectory], rewards: List[float]) -> torch.Tensor:
        losses = []
        for t, r in zip(batch_trajs, rewards):
            losses.append(self.trajectory_balance_loss(t, r))
        return torch.stack(losses).mean()



class MixedRewardController:
    """Alpha-annealed hybrid reward controller (QSAR + Docking + optional QED/SA).

    Usage:
        rc = MixedRewardController(
            qsar_checkpoint="checkpoints/qsar.pt",
            device=device,
            alpha_start=0.8,
            alpha_end=0.2,
            total_steps=10000,
            add_qed=0.1,
            sub_sa=0.05,
            dock_temp=1.0,
            # Optional: enable Plantain as the primary model term
            use_plantain=True,
            plantain_pocket_pdb="test/2y9x/2y9x_pocket.pdb",
            plantain_device="auto",
        )

        # Inside training loop per sampled terminal product 'smiles_b':
        R = rc.get_reward(smiles_b, protein_seq)
        rc.step()  # advance global step for annealing
    """

    def __init__(
        self,
        qsar_checkpoint: str | None,
        *,
        device: torch.device,
        alpha_start: float = 0.8,
        alpha_end: float = 0.2,
        total_steps: int = 10000,
        add_qed: float = 0.2,
        sub_sa: float = 0.05,
        dock_temp: float = 1.0,
        lipinski_penalty: float = 0.0,
        # Plantain options
        use_plantain: bool = False,
        plantain_pocket_pdb: str | None = None,
        plantain_device: str = "auto",
        plantain_scale: float = 10.0,
        plantain_poses_dir: str = "runs/plantain_poses_tb",
        # Vina refine options (Plantain+Vina for reward)
        use_vina: bool = False,
        vina_pocket_pdb: str | None = None,
        vina_center: list | None = None,
        vina_box_size: float = 22.0,
        vina_exhaustiveness: int = 32,
        vina_top_k: int = 1,
        vina_full_dock_th: float = -3.0,
        vina_obabel_bin: str = "/usr/local/bin/obabel",
        vina_strict: bool = False,
        vina_pdbqt_dir: str = "runs/vina_pdbqt",
        vina_weight: float = 1.0,
        vina_reward_smooth: float = 0.0,
    ) -> None:
        self.qsar = None
        if not use_plantain and isinstance(qsar_checkpoint, str) and len(qsar_checkpoint) > 0:
            try:
                self.qsar = QSARPredictor(qsar_checkpoint, device=device)
            except Exception:
                self.qsar = None
        self.device = device
        self.alpha_start = float(alpha_start)
        self.alpha_end = float(alpha_end)
        self.total_steps = max(1, int(total_steps))
        self.add_qed = float(add_qed)
        self.sub_sa = float(sub_sa)
        self.dock_temp = float(dock_temp)
        self.lipinski_penalty = float(lipinski_penalty)
        self._step = 0
        # Plantain fields
        self.use_plantain = bool(use_plantain)
        self.plantain_pocket_pdb = plantain_pocket_pdb if isinstance(plantain_pocket_pdb, str) and plantain_pocket_pdb else None
        self.plantain_device = str(plantain_device or "auto")
        self.plantain_scale = float(plantain_scale)
        self.plantain_poses_dir = str(plantain_poses_dir or "runs/plantain_poses_tb")
        # Vina settings
        self.use_vina = bool(use_vina)
        self.vina_pocket_pdb = vina_pocket_pdb if isinstance(vina_pocket_pdb, str) and vina_pocket_pdb else None
        self.vina_center = vina_center if isinstance(vina_center, list) else None
        self.vina_box_size = float(vina_box_size)
        self.vina_exhaustiveness = int(vina_exhaustiveness)
        self.vina_top_k = int(max(1, vina_top_k))
        self.vina_full_dock_th = float(vina_full_dock_th)
        self.vina_obabel_bin = str(vina_obabel_bin or "/usr/local/bin/obabel")
        self.vina_strict = bool(vina_strict)
        self.vina_pdbqt_dir = str(vina_pdbqt_dir or "runs/vina_pdbqt")
        # Weighting for docking energy contribution (reward uses -vina_weight * E)
        self.vina_weight = float(vina_weight)
        # Reward smoothing (EMA on Vina energy); 0 disables
        self.vina_reward_smooth = max(0.0, min(0.999, float(vina_reward_smooth)))
        self._ema_vina_energy: Optional[float] = None
        # Last computed vina energies for logging (updated per get_reward call)
        self.last_vina_energy: Optional[float] = None
        self.last_vina_raw_energy: Optional[float] = None
        # Initialize Vina engine/grid lazily
        self._vina = None
        if self.use_vina and isinstance(self.vina_pocket_pdb, str) and self.vina_pocket_pdb:
            try:
                from vina import Vina  # type: ignore
                center = self.vina_center
                if center is None:
                    center = self._pocket_center_from_pdb(self.vina_pocket_pdb)
                size = [max(16.0, min(60.0, self.vina_box_size))]*3
                rec = self._prepare_receptor(self.vina_pocket_pdb)
                v = Vina(sf_name="vina")
                v.set_receptor(rec)
                v.compute_vina_maps(center=center, box_size=size)
                self._vina = (v, rec, center, size)
                print({"vina_grid": {"center": center, "size": size}})
            except Exception:
                self._vina = None

    def _current_alpha(self) -> float:
        t = max(0.0, min(1.0, self._step / float(self.total_steps)))
        return self.alpha_start + (self.alpha_end - self.alpha_start) * t

    def get_reward(self, smiles_b: str, protein_seq: str) -> float:
        # Prefer Vina reward if enabled
        # Reset last energies for this call
        self.last_vina_energy = None
        self.last_vina_raw_energy = None
        if self.use_vina and self._vina is not None:
            try:
                v, rec, center, size = self._vina
                lig_pdbqt = self._meeko_pdbqt_from_smiles_via_plantain(smiles_b)
                if lig_pdbqt and os.path.exists(lig_pdbqt) and os.path.getsize(lig_pdbqt) > 0:
                    v.set_ligand_from_file(lig_pdbqt)
                    res = v.score()
                    try:
                        raw = float(res)
                    except Exception:
                        import numpy as _np  # type: ignore
                        raw = float(_np.asarray(res).ravel()[0])
                    try:
                        v.optimize()
                    except Exception:
                        pass
                    res2 = v.score()
                    try:
                        opt = float(res2)
                    except Exception:
                        import numpy as _np2  # type: ignore
                        opt = float(_np2.asarray(res2).ravel()[0])
                    if opt is None or (opt is not None and opt > self.vina_full_dock_th):
                        try:
                            v.dock(exhaustiveness=int(max(8, self.vina_exhaustiveness)), n_poses=1)
                            res3 = v.score()
                            try:
                                opt2 = float(res3)
                            except Exception:
                                import numpy as _np3  # type: ignore
                                opt2 = float(_np3.asarray(res3).ravel()[0])
                            if opt is None or (opt2 is not None and opt2 < opt):
                                opt = opt2
                        except Exception:
                            pass
                    energy = opt if opt is not None else raw
                    # Save for external logging
                    try:
                        self.last_vina_raw_energy = float(raw)
                        # Apply EMA smoothing if enabled
                        if energy is not None and self.vina_reward_smooth > 0.0:
                            if self._ema_vina_energy is None:
                                self._ema_vina_energy = float(energy)
                            else:
                                alpha = float(self.vina_reward_smooth)
                                self._ema_vina_energy = alpha * float(self._ema_vina_energy) + (1.0 - alpha) * float(energy)
                            self.last_vina_energy = float(self._ema_vina_energy)
                        else:
                            self.last_vina_energy = float(energy) if energy is not None else None
                    except Exception:
                        self.last_vina_raw_energy = None
                        self.last_vina_energy = None
                    # Use smoothed energy if present
                    energy_used = self.last_vina_energy if (self.vina_reward_smooth > 0.0 and self.last_vina_energy is not None) else energy
                    if energy_used is None:
                        return 0.0
                    # Map energy to reward (negative energy -> higher reward) and add QED/SA/Lipinski shaping
                    try:
                        qed = oracle.calculate_qed(smiles_b)
                    except Exception:
                        qed = 0.0
                    try:
                        sa = oracle.calculate_sascore(smiles_b)
                    except Exception:
                        sa = 5.0
                    lip_pen = 0.0
                    try:
                        m = Chem.MolFromSmiles(smiles_b)
                        if m is not None:
                            from rdkit.Chem import Descriptors, Crippen, Lipinski  # type: ignore
                            viol = 0
                            if Descriptors.MolWt(m) > 500: viol += 1
                            if Crippen.MolLogP(m) > 5: viol += 1
                            if Lipinski.NumHDonors(m) > 5: viol += 1
                            if Lipinski.NumHAcceptors(m) > 10: viol += 1
                            lip_pen = float(viol)
                    except Exception:
                        lip_pen = 0.0
                    base = -self.vina_weight * float(energy_used)
                    reward = max(0.0, base + self.add_qed * float(qed) - self.sub_sa * float(sa) - self.lipinski_penalty * lip_pen)
                    return reward
            except Exception:
                pass
        # Fallback to existing QSAR/Plantain hybrid
        alpha = self._current_alpha()
        return oracle.get_reward(
            smiles_b,
            protein_seq,
            use_qsar=(self.qsar is not None and not self.use_plantain),
            qsar_predict=(self.qsar.predict_pactivity if self.qsar is not None else None),
            alpha=alpha,
            dock_temp=self.dock_temp,
            add_qed=self.add_qed,
            sub_sa=self.sub_sa,
            lipinski_penalty=self.lipinski_penalty,
            use_plantain=self.use_plantain,
            plantain_pocket_pdb=self.plantain_pocket_pdb,
            plantain_device=self.plantain_device,
            plantain_scale=self.plantain_scale,
        )

    # --- Vina helpers ---
    def _pocket_center_from_pdb(self, pocket_pdb: str) -> list:
        (px, py, pz) = self._bbox_from_pdb_like(pocket_pdb)
        return [(px[0]+px[1])/2, (py[0]+py[1])/2, (pz[0]+pz[1])/2]

    def _bbox_from_pdb_like(self, path: str):
        xs = []; ys = []; zs = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    if ln.startswith("ATOM") or ln.startswith("HETATM"):
                        try:
                            x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
                        except Exception:
                            parts = ln.split()
                            if len(parts) < 9:
                                continue
                            x = float(parts[-6]); y = float(parts[-5]); z = float(parts[-4])
                        xs.append(x); ys.append(y); zs.append(z)
            return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))
        except Exception:
            return ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

    def _prepare_receptor(self, pocket_pdb: str) -> str:
        rec_fix = pocket_pdb.replace(".pdb", "_rigid.pdbqt")
        for prep_cmd in ("prepare_receptor", "prepare_receptor4.py"):
            try:
                r = __import__("subprocess").run([prep_cmd, "-r", pocket_pdb, "-o", rec_fix, "-A", "checkhydrogens"], stdout=__import__("subprocess").PIPE, stderr=__import__("subprocess").STDOUT, text=True)
                if r.returncode == 0 and os.path.exists(rec_fix) and os.path.getsize(rec_fix) > 0:
                    print({"vina_receptor": {"method": prep_cmd, "path": rec_fix}})
                    return rec_fix
            except FileNotFoundError:
                continue
            except Exception:
                continue
        # Require ADFRtools for receptor preparation; no OpenBabel fallback for receptors
        raise RuntimeError("ADFR prepare_receptor not available; ensure ADFRtools is installed and 'prepare_receptor' is in PATH")

    def _meeko_pdbqt_from_smiles_via_plantain(self, smi: str) -> Optional[str]:
        # Use PLANTAIN to generate a pose SDF, then Meeko to PDBQT (with robust cwd, config, and fallbacks)
        if not (isinstance(self.plantain_pocket_pdb, str) and self.plantain_pocket_pdb):
            return None
        try:
            plant_dir = os.path.join(os.path.dirname(__file__), "..", "lib", "plantain")
            plant_dir = os.path.abspath(plant_dir)
            if plant_dir not in sys.path:
                sys.path.insert(0, plant_dir)
            from common.cfg_utils import get_config  # type: ignore
            from models.pretrained_plantain import get_pretrained_plantain  # type: ignore
            from datasets.inference_dataset import InferenceDataset  # type: ignore
            from terrace import collate  # type: ignore
            from rdkit import Chem  # type: ignore
            from common.pose_transform import add_multi_pose_to_mol  # type: ignore
            # Ensure Plantain runs under its repo cwd so relative configs resolve
            _old = os.getcwd()
            try:
                os.chdir(plant_dir)
                cfg = get_config("icml", folder=os.path.join(plant_dir, "configs"))
                # Disable torch.compile for compatibility
                try:
                    cfg.platform["compile"] = False  # type: ignore[index]
                except Exception:
                    try:
                        setattr(cfg.platform, "compile", False)
                    except Exception:
                        pass
                model = get_pretrained_plantain()
                try:
                    model.eval()
                except Exception:
                    pass
                import tempfile as _tmp
                td = _tmp.mkdtemp(prefix="tb_vina_pose_")
                smi_path = os.path.join(td, "one.smi"); open(smi_path, "w", encoding="utf-8").write(smi+"\n")
                ds = InferenceDataset(cfg, smi_path, self.plantain_pocket_pdb, model.get_input_feats())
                if len(ds) <= 0:
                    print({"plantain_error": "empty_dataset"})
                    return None
                x, y = ds[0]
                batch = collate([x])
                try:
                    dev = (self.plantain_device or "auto").lower()
                    if dev == "auto":
                        import torch
                        if torch.cuda.is_available():
                            dev = "cuda:0"
                        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                            dev = "mps"
                        else:
                            dev = "cpu"
                    batch = batch.to(dev); model = model.to(dev)
                except Exception:
                    pass
                try:
                    pred = model(batch)[0]
                except Exception as e_inf:
                    print({"plantain_infer_error": str(e_inf)})
                    return None
                mol = getattr(x, "lig", None)
                if mol is None or not hasattr(pred, "lig_pose") or pred.lig_pose is None:
                    print({"plantain_error": "no_lig_pose"})
                    return None
                add_multi_pose_to_mol(mol, pred.lig_pose)
                # Resolve poses_dir relative to project root if not absolute
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                poses_dir = self.plantain_poses_dir
                if not os.path.isabs(poses_dir):
                    poses_dir = os.path.join(project_root, poses_dir)
                os.makedirs(poses_dir, exist_ok=True)
                sdf_path = os.path.join(poses_dir, f"{abs(hash(smi))}.sdf")
                w = Chem.SDWriter(sdf_path); w.write(mol, confId=0); w.close()
                print({"plantain_pose_sdf": sdf_path})
            finally:
                try:
                    os.chdir(_old)
                except Exception:
                    pass
            # Meeko
            from meeko import MoleculePreparation, PDBQTWriterLegacy  # type: ignore
            mols = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)
            mol2 = None
            for _m in mols:
                if _m is not None:
                    mol2 = _m
                    break
            if mol2 is None:
                print({"meeko_error": "sdf_read_failed", "sdf": sdf_path})
                return None
            try:
                mol2 = Chem.AddHs(mol2, addCoords=True)
            except Exception:
                mol2 = Chem.AddHs(mol2)
            prep = MoleculePreparation(); u = prep.prepare(mol2)
            if isinstance(u, (list, tuple)):
                u = u[0]
            s = PDBQTWriterLegacy().write_string(u, bad_charge_ok=True)
            if isinstance(s, tuple):
                s = s[0]
            if not s or not str(s).strip():
                print({"meeko_pdbqt_empty_string": True, "smiles": smi[:32]})
                return None
            os.makedirs(self.vina_pdbqt_dir, exist_ok=True)
            out = os.path.join(self.vina_pdbqt_dir, f"{abs(hash(smi))}.pdbqt")
            # Cache reuse
            try:
                if os.path.exists(out) and os.path.getsize(out) > 0:
                    return out
            except Exception:
                pass
            with open(out, "w", encoding="utf-8") as _f:
                _f.write(s)
            try:
                if os.path.getsize(out) <= 0:
                    print({"meeko_pdbqt_empty_file": out})
                    return None
            except Exception:
                return None
            print({"meeko_pdbqt": out})
            return out
        except Exception as e_outer:
            print({"plantain_outer_error": str(e_outer)})
            if self.vina_strict:
                return None
            # As a last resort, build a simple 3D conformer and export via obabel
            try:
                from rdkit import Chem  # type: ignore
                from rdkit.Chem import AllChem  # type: ignore
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    return None
                m = Chem.AddHs(m); AllChem.EmbedMolecule(m, AllChem.ETKDG())
                # Resolve poses_dir for fallback as well
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                poses_dir = self.plantain_poses_dir
                if not os.path.isabs(poses_dir):
                    poses_dir = os.path.join(project_root, poses_dir)
                os.makedirs(poses_dir, exist_ok=True)
                sdf = os.path.join(poses_dir, f"{abs(hash(smi))}_fallback.sdf")
                from rdkit.Chem import SDWriter
                SDWriter(sdf).write(m)
                out = os.path.join(self.vina_pdbqt_dir, f"{abs(hash(smi))}.pdbqt")
                os.makedirs(self.vina_pdbqt_dir, exist_ok=True)
                __import__("subprocess").run([self.vina_obabel_bin, "-isdf", sdf, "-opdbqt", "-O", out, "-h"], check=True)
                print({"meeko_fallback_smiles3d_pdbqt": out})
                try:
                    if os.path.exists(out) and os.path.getsize(out) > 0:
                        return out
                except Exception:
                    pass
                return None
            except Exception:
                return None

    def step(self, n: int = 1) -> None:
        self._step += max(0, int(n))

