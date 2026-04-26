from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch_geometric.data import Data

from rdkit import Chem


# ---------------------------
# RDKit helpers
# ---------------------------


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def canonical_smiles(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol)


# ---------------------------
# Atom featurization
# ---------------------------


_ELEMENTS = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}

_DEGREES = list(range(0, 6))
_FORMAL_CHARGES = [-2, -1, 0, 1, 2]
_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]


def _one_hot(value, choices) -> List[int]:
    return [1 if value == c else 0 for c in choices]


def get_atom_feature_dim() -> int:
    # elements + other
    num_elem = len(_ELEMENTS) + 1
    num_deg = len(_DEGREES) + 1
    num_charge = len(_FORMAL_CHARGES) + 1
    num_hyb = len(_HYBRIDIZATIONS) + 1
    num_bool = 2  # aromatic
    return num_elem + num_deg + num_charge + num_hyb + num_bool


def atom_features(atom: Chem.Atom) -> List[float]:
    z = atom.GetAtomicNum()
    elem_idx = z if z in _ELEMENTS else None
    elem_oh = _one_hot(z if z in _ELEMENTS else "other", list(_ELEMENTS.keys()) + ["other"])

    deg = atom.GetDegree()
    deg_oh = _one_hot(deg if deg in _DEGREES else "other", _DEGREES + ["other"])

    chg = atom.GetFormalCharge()
    chg_oh = _one_hot(chg if chg in _FORMAL_CHARGES else "other", _FORMAL_CHARGES + ["other"])

    hyb = atom.GetHybridization()
    hyb_oh = _one_hot(hyb if hyb in _HYBRIDIZATIONS else "other", _HYBRIDIZATIONS + ["other"])

    aromatic = 1 if atom.GetIsAromatic() else 0

    feat = elem_oh + deg_oh + chg_oh + hyb_oh + [aromatic, 1 - aromatic]
    return [float(x) for x in feat]


def build_graph_from_mol(mol: Chem.Mol) -> Optional[Data]:
    if mol is None:
        return None
    try:
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return None
        # Node features
        x = torch.tensor([atom_features(mol.GetAtomWithIdx(i)) for i in range(num_atoms)], dtype=torch.float32)

        # Edge index (undirected, both directions)
        edges: List[Tuple[int, int]] = []
        for bond in mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            edges.append((a, b))
            edges.append((b, a))
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            # Molecule with single atom
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)
    except Exception:
        return None


def build_graph_from_smiles(smiles: str) -> Optional[Data]:
    mol = mol_from_smiles(smiles)
    return build_graph_from_mol(mol)


@dataclass
class GraphBuildResult:
    data: Optional[Data]
    smiles: str
    valid: bool


