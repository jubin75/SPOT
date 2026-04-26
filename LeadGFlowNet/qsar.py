from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem

from LeadGFlowNet.protein_encoder import SimpleProteinEncoder, tokenize_protein


def morgan_fp_bits(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    # RDKit explicit conversion
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)


class ProteinLigandDataset(Dataset):
    def __init__(self, csv_path: str, n_bits: int = 2048, radius: int = 2):
        import pandas as pd
        df = pd.read_csv(csv_path)
        # Expect columns: protein_sequence, ligand_smiles, p_activity
        for col in ["protein_sequence", "ligand_smiles", "p_activity"]:
            if col not in df.columns:
                raise ValueError(f"CSV missing column: {col}")
        self.proteins: List[str] = df["protein_sequence"].astype(str).tolist()
        self.smiles: List[str] = df["ligand_smiles"].astype(str).tolist()
        self.pact: List[float] = df["p_activity"].astype(float).tolist()
        self.n_bits = int(n_bits)
        self.radius = int(radius)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        prot = self.proteins[idx]
        smi = self.smiles[idx]
        y = float(self.pact[idx])
        fp = morgan_fp_bits(smi, n_bits=self.n_bits, radius=self.radius)
        return prot, fp, y


class QSARMLP(nn.Module):
    def __init__(self, fp_dim: int, protein_embed_dim: int = 256, lstm_hidden: int = 256, hidden: int = 512):
        super().__init__()
        self.prot_enc = SimpleProteinEncoder(embed_dim=protein_embed_dim, lstm_hidden=lstm_hidden)
        self.fp_dim = int(fp_dim)
        self.out_dim = self.prot_enc.out_dim
        in_dim = self.fp_dim + self.out_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, token_ids: torch.Tensor, fp_tensor: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, L) or (L,), fp_tensor: (B, fp_dim)
        prot_h = self.prot_enc(token_ids)
        if fp_tensor.dim() == 1:
            fp_tensor = fp_tensor.unsqueeze(0)
        x = torch.cat([fp_tensor, prot_h], dim=1)
        y = self.mlp(x).squeeze(-1)
        return y


@dataclass
class QSARConfig:
    n_bits: int = 2048
    radius: int = 2
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 5
    device: str = "cpu"


def _collate(batch: List[Tuple[str, np.ndarray, float]] | Tuple[List[str], np.ndarray | torch.Tensor, List[float] | np.ndarray | torch.Tensor], device: torch.device):
    """Collate function tolerant to default DataLoader collation and raw list-of-samples.

    Accepts either:
      - list of (protein:str, fp:np.ndarray, y:float)
      - tuple (list[str], Tensor/ndarray, list/ndarray/Tensor)
    """
    # Detect already-collated (tuple of length 3 where first element is list of strings)
    if isinstance(batch, (list, tuple)) and len(batch) == 3 and isinstance(batch[0], list):
        prots, fps, ys = batch  # type: ignore[assignment]
    else:
        # Expect list of triples
        prots, fps, ys = zip(*batch)  # type: ignore[misc]

    # Protein tokenization (pad to max length in batch)
    token_list = [tokenize_protein(p) for p in prots]
    max_len = max(t.numel() for t in token_list) if token_list else 1
    tokens = []
    for t in token_list:
        if t.numel() < max_len:
            pad = torch.zeros((max_len - t.numel(),), dtype=torch.long)
            t = torch.cat([t, pad], dim=0)
        tokens.append(t)
    token_ids = torch.stack(tokens, dim=0).to(device) if tokens else torch.zeros((0, max_len), dtype=torch.long, device=device)

    # Fingerprints to Tensor
    if isinstance(fps, torch.Tensor):
        fp_tensor = fps.to(device=device, dtype=torch.float32)
    elif isinstance(fps, np.ndarray):
        fp_tensor = torch.tensor(fps, dtype=torch.float32, device=device)
    else:
        # list of arrays
        fp_tensor = torch.tensor(np.stack(fps, axis=0), dtype=torch.float32, device=device)

    # Targets to Tensor
    if isinstance(ys, torch.Tensor):
        y_tensor = ys.to(device=device, dtype=torch.float32)
    elif isinstance(ys, np.ndarray):
        y_tensor = torch.tensor(ys, dtype=torch.float32, device=device)
    else:
        y_tensor = torch.tensor(list(ys), dtype=torch.float32, device=device)

    return token_ids, fp_tensor, y_tensor


def train_qsar(csv_path: str, save_path: str = "checkpoints/qsar.pt", cfg: Optional[QSARConfig] = None) -> str:
    cfg = cfg or QSARConfig()
    device = torch.device(cfg.device)
    ds = ProteinLigandDataset(csv_path, n_bits=cfg.n_bits, radius=cfg.radius)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda b: _collate(b, device),
    )

    model = QSARMLP(fp_dim=cfg.n_bits).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        total = 0.0
        count = 0
        for token_ids, fp_tensor, y_tensor in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(token_ids, fp_tensor)
            loss = loss_fn(pred, y_tensor)
            loss.backward()
            opt.step()
            total += loss.item() * y_tensor.numel()
            count += y_tensor.numel()
        print({"epoch": epoch, "mse": total / max(1, count)})

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "state": model.state_dict(),
        "n_bits": cfg.n_bits,
        "radius": cfg.radius,
    }, save_path)
    print({"saved": save_path})
    return save_path


class QSARPredictor:
    def __init__(self, checkpoint_path: str, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cpu")
        obj = torch.load(checkpoint_path, map_location=device)
        self.n_bits = int(obj.get("n_bits", 2048))
        self.radius = int(obj.get("radius", 2))
        self.model = QSARMLP(fp_dim=self.n_bits).to(device)
        self.model.load_state_dict(obj["state"], strict=False)
        self.model.eval()
        self.device = device
        # Simple caches to avoid repeated work across many calls
        self._protein_tokens_cache: dict[str, torch.Tensor] = {}
        self._qsar_cache: dict[tuple[str, str], float] = {}

    @torch.no_grad()
    def predict_pactivity(self, smiles: str, protein_seq: str) -> float:
        key = (smiles, protein_seq)
        if key in self._qsar_cache:
            return self._qsar_cache[key]
        fp = morgan_fp_bits(smiles, n_bits=self.n_bits, radius=self.radius)
        fp_tensor = torch.tensor(fp, dtype=torch.float32, device=self.device)
        token_ids = self._get_protein_tokens(protein_seq)
        y = self.model(token_ids, fp_tensor)
        out = float(y.item())
        self._qsar_cache[key] = out
        return out

    @torch.no_grad()
    def predict_pactivity_batch(self, smiles_list: List[str], protein_seq: str, batch_size: int = 512) -> List[float]:
        """Batch score a list of SMILES for a single protein sequence.

        Returns a list of raw predictions (before sigmoid), same order as input.
        Results are cached per (smiles, protein_seq).
        """
        results: List[float] = [0.0] * len(smiles_list)
        # Check cache first
        to_compute_indices: List[int] = []
        for i, s in enumerate(smiles_list):
            key = (s, protein_seq)
            if key in self._qsar_cache:
                results[i] = self._qsar_cache[key]
            else:
                to_compute_indices.append(i)
        if not to_compute_indices:
            return results
        # Prepare protein tokens once and repeat in batches
        token_ids_full = self._get_protein_tokens(protein_seq)
        # Build fingerprints for remaining items (RDKit is CPU-bound; do in Python loop, then batch to model)
        fps_all: List[np.ndarray] = []
        for i in to_compute_indices:
            s = smiles_list[i]
            fp = morgan_fp_bits(s, n_bits=self.n_bits, radius=self.radius)
            fps_all.append(fp)
        # Convert to tensor on device in chunks to control memory
        start = 0
        while start < len(to_compute_indices):
            end = min(start + int(batch_size), len(to_compute_indices))
            idx_slice = to_compute_indices[start:end]
            fp_tensor = torch.tensor(np.stack(fps_all[start:end], axis=0), dtype=torch.float32, device=self.device)
            # Repeat protein tokens to match batch
            if token_ids_full.dim() == 1:
                tok_rep = token_ids_full.unsqueeze(0).repeat(fp_tensor.size(0), 1)
            else:
                tok_rep = token_ids_full
                if tok_rep.size(0) != fp_tensor.size(0):
                    tok_rep = tok_rep[0:1, :].repeat(fp_tensor.size(0), 1)
            y = self.model(tok_rep, fp_tensor).detach().cpu().numpy().tolist()
            for j, out in enumerate(y):
                idx = idx_slice[j]
                s = smiles_list[idx]
                results[idx] = float(out)
                self._qsar_cache[(s, protein_seq)] = float(out)
            start = end
        return results

    def _get_protein_tokens(self, protein_seq: str) -> torch.Tensor:
        if protein_seq in self._protein_tokens_cache:
            t = self._protein_tokens_cache[protein_seq]
            return t
        token_ids = tokenize_protein(protein_seq).to(self.device)
        self._protein_tokens_cache[protein_seq] = token_ids
        return token_ids


