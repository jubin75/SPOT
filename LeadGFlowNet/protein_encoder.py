from __future__ import annotations

from typing import List
import os

import torch
from torch import nn


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # 0 is pad/unk


def tokenize_protein(sequence: str, max_len: int = 1024) -> torch.Tensor:
    seq = (sequence or "").upper()[:max_len]
    ids: List[int] = [AA_TO_IDX.get(ch, 0) for ch in seq]
    if not ids:
        ids = [0]
    return torch.tensor(ids, dtype=torch.long)


class SimpleProteinEncoder(nn.Module):
    """A light-weight protein encoder (embedding + 1-layer BiLSTM + mean pool).

    Replaceable with ESM/ProtBERT embeddings without changing downstream code.
    """

    def __init__(self, embed_dim: int = 256, lstm_hidden: int = 256):
        super().__init__()
        vocab_size = len(AMINO_ACIDS) + 1
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.out_dim = lstm_hidden * 2

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, L) or (L,) -> returns (B, out_dim)"""
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        x = self.embed(token_ids)
        x, _ = self.lstm(x)
        h = x.mean(dim=1)
        return h



class Esm2ProteinEncoder(nn.Module):
    """ESM-2 encoder via Hugging Face transformers.

    Requires the 'transformers' package and a valid ESM2 checkpoint name, e.g.:
      - facebook/esm2_t6_8M_UR50D (fast, small)
      - facebook/esm2_t12_35M_UR50D
      - facebook/esm2_t30_150M_UR50D
      - facebook/esm2_t33_650M_UR50D (largest)

    Usage mirrors SimpleProteinEncoder from the outside via:
      - .out_dim: hidden size
      - encode_sequence(sequence: str) -> Tensor[(1, out_dim)]
    """

    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D"):
        super().__init__()
        # Lazy import to avoid hard dependency unless used
        try:
            from transformers import AutoTokenizer, EsmModel  # type: ignore
        except Exception as e:
            raise ImportError(
                "transformers is required for Esm2ProteinEncoder. Install via 'pip install transformers'."
            ) from e

        self._model_name = str(model_name)
        self._AutoTokenizer = AutoTokenizer
        self._EsmModel = EsmModel

        # Resolve local snapshot in lib/ if present
        resolved_name = self._resolve_local_snapshot(self._model_name)

        # Initialize tokenizer and model (graceful offline fallback)
        try:
            local_only = os.path.isdir(resolved_name)
            self.tokenizer = self._AutoTokenizer.from_pretrained(
                resolved_name, do_lower_case=False, local_files_only=local_only
            )
            self.model = self._EsmModel.from_pretrained(
                resolved_name, local_files_only=local_only, ignore_mismatched_sizes=True
            )
        except Exception as e:
            # Likely offline or repo not reachable
            raise OSError(
                f"Failed to load ESM2 model '{self._model_name}'. If running offline, either pre-cache the model, pass a local path via --esm2-model, or set --protein-encoder simple. Original error: {e}"
            )
        self.model.eval()
        self.out_dim = int(self.model.config.hidden_size)

    @staticmethod
    def _resolve_local_snapshot(model_name: str) -> str:
        """If a local snapshot exists in project lib/, return that path.

        Accepts either a HF repo id (e.g., 'facebook/esm2_t30_150M_UR50D') or a local directory path.
        If given the repo id and lib/models--facebook--esm2_t30_150M_UR50D exists, pick latest snapshot.
        If given a directory that contains 'snapshots', pick the first subdir.
        Otherwise, return the original string.
        """
        # If user already provided a valid directory, potentially normalize to snapshot
        if os.path.isdir(model_name):
            snap_dir = os.path.join(model_name, "snapshots")
            if os.path.isdir(snap_dir):
                try:
                    candidates = [os.path.join(snap_dir, d) for d in os.listdir(snap_dir)]
                    candidates = [p for p in candidates if os.path.isdir(p)]
                    if candidates:
                        # Pick the most recently modified snapshot
                        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                        return candidates[0]
                except Exception:
                    pass
            return model_name

        # Attempt to find local copy inside project lib/
        try:
            here = os.path.abspath(os.path.dirname(__file__))
            root = os.path.abspath(os.path.join(here, ".."))
            lib_dir = os.path.join(root, "lib")
            # Map known HF id to lib cache folder name
            if model_name.startswith("facebook/esm2_t"):
                cache_root = os.path.join(lib_dir, "models--facebook--esm2_t30_150M_UR50D") if "t30_150M" in model_name else None
                if cache_root and os.path.isdir(cache_root):
                    snap_dir = os.path.join(cache_root, "snapshots")
                    if os.path.isdir(snap_dir):
                        snaps = [os.path.join(snap_dir, d) for d in os.listdir(snap_dir)]
                        snaps = [p for p in snaps if os.path.isdir(p)]
                        if snaps:
                            snaps.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                            return snaps[0]
        except Exception:
            pass
        return model_name

    @torch.no_grad()
    def encode_sequence(self, sequence: str, *, device: torch.device | None = None) -> torch.Tensor:
        # Prepare inputs
        inputs = self.tokenizer(sequence or "", return_tensors="pt", add_special_tokens=True)
        if device is None:
            device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state  # (1, L, H)
        mask = inputs.get("attention_mask")
        if mask is None:
            # Fallback: simple mean pool
            pooled = hidden.mean(dim=1)
        else:
            mask_f = mask.unsqueeze(-1).to(hidden.dtype)  # (1, L, 1)
            summed = (hidden * mask_f).sum(dim=1)
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            pooled = summed / denom
        return pooled

