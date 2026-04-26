from __future__ import annotations

from typing import Optional

import math
import os
from functools import lru_cache


class _Chdir:
    """Context manager to temporarily change the working directory."""
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


@lru_cache(maxsize=1)
def _load_plantain_model(device: str = "cpu"):
    """Lazy-load PLANTAIN pretrained model and return (model, cfg).

    We temporarily chdir into lib/plantain so relative paths in that repo
    (configs/, data/plantain_final.pt) resolve correctly.
    """
    plantain_dir = _get_plantain_dir()
    try:
        import sys
        if plantain_dir not in sys.path:
            sys.path.insert(0, plantain_dir)
        with _Chdir(plantain_dir):
            from common.cfg_utils import get_config  # type: ignore
            from models.pretrained_plantain import get_pretrained_plantain  # type: ignore
            cfg = get_config("icml")
            model = get_pretrained_plantain()
            try:
                import torch  # type: ignore
                model = model.to(device)
            except Exception:
                pass
            model.eval()
            return model, cfg
    except Exception:
        return None, None


def _plantain_min_score_for_smiles(smiles_b: str, pocket_pdb_path: str, device: str = "cpu") -> Optional[float]:
    """Return the best (minimum) PLANTAIN score for one SMILES against a pocket PDB, or None on failure."""
    plantain_dir = _get_plantain_dir()
    model, cfg = _load_plantain_model(device)
    if model is None or cfg is None:
        return None
    if not os.path.exists(pocket_pdb_path):
        return None
    try:
        import sys
        if plantain_dir not in sys.path:
            sys.path.insert(0, plantain_dir)
        with _Chdir(plantain_dir):
            from datasets.inference_dataset import InferenceDataset  # type: ignore
            from terrace import collate  # type: ignore
            import tempfile
            # Write a temporary .smi file containing this single SMILES
            with tempfile.TemporaryDirectory() as td:
                smi_path = os.path.join(td, "one.smi")
                with open(smi_path, "w", encoding="utf-8") as f:
                    f.write(str(smiles_b).strip() + "\n")
                dataset = InferenceDataset(cfg, smi_path, pocket_pdb_path, model.get_input_feats())
                if len(dataset) <= 0:
                    return None
                x, y = dataset[0]
                # Build one-sample batch on device if torch available
                try:
                    import torch  # type: ignore
                    batch = collate([x])
                    # Move lazy tensors inside the batch to device if possible
                    try:
                        batch = batch.to(device)
                    except Exception:
                        pass
                    pred = model(batch)[0]
                    # pred.score is sorted ascending (lower is better)
                    if hasattr(pred, "score") and getattr(pred, "score") is not None:
                        s0 = float(pred.score[0].detach().cpu().item())
                    else:
                        s0 = None
                except Exception:
                    return None
                return s0
    except Exception:
        return None

# Docking is optional. For RL we default to QSAR-only; docking requires PDBQT prep.
def run_vina_docking(smiles_b: str, protein_seq: str, vina_calculator: Optional[object] = None) -> float:
    """Return docking score (lower-is-better). If no calculator provided, return 0.0.

    Provide a `vina_calculator` with method `.get_reward(smiles: str) -> float` that returns
    a positive reward so we invert here to a score if needed.
    """
    try:
        if vina_calculator is not None and hasattr(vina_calculator, "get_reward"):
            # Many docking tools output negative scores (lower better). Users can pass either
            # a score or reward; we conservatively treat returned value as a reward and invert.
            reward = float(vina_calculator.get_reward(smiles_b))
            return -reward
    except Exception:
        pass
    return 0.0

def calculate_qed(smiles_b: str) -> float:
    """RDKit QED score in [0,1]; fallback to 0 on failure."""
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import QED  # type: ignore
        mol = Chem.MolFromSmiles(smiles_b)
        if mol is None:
            return 0.0
        return float(QED.qed(mol))
    except Exception:
        return 0.0

def calculate_sascore(smiles_b: str) -> float:
    """Synthetic Accessibility (lower is better). Returns 10 on failure (hard)."""
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import rdMolDescriptors as rdmd  # type: ignore
        mol = Chem.MolFromSmiles(smiles_b)
        if mol is None:
            return 10.0
        # Heuristic SA proxy: larger, more rings/hetero atoms -> higher SA
        num_rings = rdmd.CalcNumRings(mol)
        heavy = mol.GetNumHeavyAtoms()
        hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (6, 1))
        score = 1.0 + 0.1 * heavy + 0.3 * num_rings + 0.2 * hetero
        return float(min(10.0, max(1.0, score)))
    except Exception:
        return 10.0

def transform_docking(docking_score: float) -> float:
    # Convert lower-is-better to higher-is-better reward
    return -docking_score


def transform_plantain(score_min: float, *, scale: float = 10.0) -> float:
    """Map PLANTAIN minimal score (lower is better) to a positive reward in (0,1].

    Uses an exponential mapping: reward = exp(-score_min / scale).
    Choose 'scale' to control sensitivity (larger scale -> flatter curve).
    """
    try:
        s = float(score_min)
        k = max(1e-6, float(scale))
        import math as _m
        r = _m.exp(-s / k)
        # Numerically guard and clamp into [0,1]
        if not _m.isfinite(r):
            return 0.0
        return float(min(1.0, max(0.0, r)))
    except Exception:
        return 0.0

def _normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    x = (value - lo) / (hi - lo)
    return float(min(1.0, max(0.0, x)))


def pactivity_to_reward(pact: float, center: float = 5.5, scale: float = 1.0) -> float:
    # Smooth mapping: sigmoid((pAct - center)/scale)
    import math
    return 1.0 / (1.0 + math.exp(-(pact - center) / max(1e-6, scale)))


def get_reward(
    smiles_b: str,
    protein_seq: str,
    *,
    use_qsar: bool = False,
    qsar_predict=None,
    alpha: float = 0.7,
    use_docking: bool = False,
    dock_temp: float = 1.0,
    add_qed: float = 0.2,
    sub_sa: float = 0.05,
    lipinski_penalty: float = 0.0,
    vina_calculator: Optional[object] = None,
    # PLANTAIN integration
    use_plantain: bool = False,
    plantain_pocket_pdb: Optional[str] = None,
    plantain_device: str = "auto",
    plantain_scale: float = 10.0,
) -> float:
    """Hybrid reward for GFlowNet.

    Args:
        use_qsar: if True, expect qsar_predict(smiles, protein)->pAct; else skip.
        alpha: mixing weight for QSAR vs docking. R = alpha*R_qsar + (1-alpha)*R_dock + add_qed*QED - sub_sa*SA.
        dock_temp: temperature for docking transform.
    """
    # Docking component (optional). Skip when alpha~1 or use_docking is False.
    if use_docking and alpha < 0.999:
        docking_score = run_vina_docking(smiles_b, protein_seq, vina_calculator)
        r_dock = transform_docking(docking_score)
    else:
        r_dock = 0.0
    # Optional QSAR or PLANTAIN component (mutually exclusive in typical usage)
    r_qsar = 0.0
    r_plant = 0.0
    if use_qsar and qsar_predict is not None and not use_plantain:
        try:
            p_act_pred = float(qsar_predict(smiles_b, protein_seq))
            r_qsar = pactivity_to_reward(p_act_pred, center=5.5, scale=0.5)
        except Exception:
            r_qsar = 0.0
    elif use_plantain and isinstance(plantain_pocket_pdb, str) and plantain_pocket_pdb:
        try:
            dev = "cuda:0"
            if plantain_device.lower() == "cpu":
                dev = "cpu"
            elif plantain_device.lower() == "mps":
                dev = "mps"
            elif plantain_device.lower() == "auto":
                # pick GPU if available else CPU
                try:
                    import torch  # type: ignore
                    if torch.cuda.is_available():
                        dev = "cuda:0"
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        dev = "mps"
                    else:
                        dev = "cpu"
                except Exception:
                    dev = "cpu"
            score_min = _plantain_min_score_for_smiles(smiles_b, plantain_pocket_pdb, device=dev)
            if score_min is not None:
                r_plant = transform_plantain(score_min, scale=float(plantain_scale))
        except Exception:
            r_plant = 0.0
    # Optional medicinal chemistry terms
    qed_score = calculate_qed(smiles_b)
    sa_score = calculate_sascore(smiles_b)

    # Optional Lipinski-like penalty (count violations among MW, LogP, HBD, HBA)
    lip_pen = 0.0
    if lipinski_penalty and lipinski_penalty > 0:
        try:
            from rdkit import Chem  # type: ignore
            from rdkit.Chem import Descriptors, Crippen, Lipinski  # type: ignore
            mol = Chem.MolFromSmiles(smiles_b)
            viol = 0
            if mol is not None:
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                hbd = Lipinski.NumHDonors(mol)
                hba = Lipinski.NumHAcceptors(mol)
                if mw > 500:
                    viol += 1
                if logp > 5:
                    viol += 1
                if hbd > 5:
                    viol += 1
                if hba > 10:
                    viol += 1
            lip_pen = float(viol)
        except Exception:
            lip_pen = 0.0

    # If PLANTAIN is enabled, treat it as the primary model component in place of QSAR
    r_primary = (r_qsar if not use_plantain else r_plant)
    reward = (
        alpha * r_primary
        + (1.0 - alpha) * r_dock
        + add_qed * qed_score
        - sub_sa * sa_score
        - lipinski_penalty * lip_pen
    )
    return max(0.0, float(reward))


def binarize_pactivity(p_activity: float, high: float = 6.0, low: float = 5.0) -> Optional[int]:
    """Binary label for offline supervised-style training.

    Returns 1 if p_activity >= high, 0 if <= low, else None (ambiguous zone).
    """
    try:
        x = float(p_activity)
    except Exception:
        return None
    if x >= high:
        return 1
    if x <= low:
        return 0
    return None

