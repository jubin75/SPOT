from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from SynthPolicyNet.data_utils import build_graph_from_smiles, get_atom_feature_dim, canonical_smiles


@dataclass(frozen=True)
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def from_tokens(cls, tokens: List[str]) -> "Vocab":
        uniq = []
        seen = set()
        for t in tokens:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        stoi = {t: i for i, t in enumerate(uniq)}
        return cls(stoi=stoi, itos=uniq)

    def to_json(self) -> str:
        return json.dumps({"itos": self.itos}, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Vocab":
        obj = json.loads(s)
        itos = obj["itos"]
        return cls(stoi={t: i for i, t in enumerate(itos)}, itos=itos)


class ForwardTrajectoryDataset(Dataset):
    """
    Dataset built from forward one-step trajectories.

    Expects columns:
      - state_smiles
      - action_building_block (single block SMILES)
      - action_reaction_template (string label)
      - is_start_state (bool); rows with True are skipped by default
      - is_in_forward_chain (bool); can be used to filter to the main chain
    """

    def __init__(
        self,
        df: pd.DataFrame,
        block_vocab: Optional[Vocab] = None,
        rxn_vocab: Optional[Vocab] = None,
        use_only_forward_chain: bool = True,
        skip_start_states: bool = True,
        min_block_freq: int = 1,
    ) -> None:
        super().__init__()
        self._raw = df.copy()

        # Filtering
        mask = pd.Series([True] * len(df))
        if use_only_forward_chain and "is_in_forward_chain" in df.columns:
            mask &= df["is_in_forward_chain"].astype(bool)
        if skip_start_states and "is_start_state" in df.columns:
            mask &= ~df["is_start_state"].astype(bool)

        # NA-like checker (mirrors forward_trajectories.is_na_like)
        def _is_na_like(value: object) -> bool:
            if value is None:
                return True
            if isinstance(value, float) and math.isnan(value):
                return True
            if isinstance(value, str):
                s = value.strip()
                if not s:
                    return True
                if s.lower() in {"n/a", "na", "nan"}:
                    return True
            return False

        # Basic field presence using NA-like semantics
        if "state_smiles" in df.columns:
            mask &= ~df["state_smiles"].apply(_is_na_like)
        if "action_building_block" in df.columns:
            mask &= ~df["action_building_block"].apply(_is_na_like)
        if "action_reaction_template" in df.columns:
            mask &= ~df["action_reaction_template"].apply(_is_na_like)

        self.df = df[mask].reset_index(drop=True)

        # Frequency pruning for building blocks
        if min_block_freq and min_block_freq > 1 and "action_building_block" in self.df.columns:
            counts = self.df["action_building_block"].value_counts()
            keep = set(counts[counts >= int(min_block_freq)].index.tolist())
            self.df = self.df[self.df["action_building_block"].isin(list(keep))].reset_index(drop=True)

        # Vocabularies
        if block_vocab is None:
            blocks = [canonical_smiles(s) for s in self.df["action_building_block"].astype(str).tolist()]
            block_vocab = Vocab.from_tokens(blocks)
        if rxn_vocab is None:
            rxns = self.df["action_reaction_template"].astype(str).tolist()
            rxn_vocab = Vocab.from_tokens(rxns)

        self.block_vocab = block_vocab
        self.rxn_vocab = rxn_vocab

        # Precompute and cache state graphs
        self._state_graphs: List[Optional[Data]] = [None] * len(self.df)

        # Build block graphs once
        self.block_graphs: List[Optional[Data]] = []
        for s in self.block_vocab.itos:
            g = build_graph_from_smiles(s)
            self.block_graphs.append(g)

        # Expose feature dim for convenience
        self.node_feature_dim: int = get_atom_feature_dim()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Data:
        row = self.df.iloc[idx]
        state_smiles = str(row["state_smiles"]) if pd.notna(row["state_smiles"]) else ""
        block_s = canonical_smiles(str(row["action_building_block"]))
        rxn_s = str(row["action_reaction_template"])

        # Graph for state
        if self._state_graphs[idx] is None:
            g = build_graph_from_smiles(state_smiles)
            if g is None:
                # Minimal single-node graph to avoid crashes; mark invalid if needed
                x = torch.zeros((1, self.node_feature_dim), dtype=torch.float32)
                g = Data(x=x, edge_index=torch.zeros((2, 0), dtype=torch.long))
            self._state_graphs[idx] = g
        data = self._state_graphs[idx].clone()

        # Labels
        y_block = self.block_vocab.stoi.get(block_s)
        if y_block is None:
            # Fallback: add on the fly (should not happen if vocab built from dataset)
            y_block = 0
        y_rxn = self.rxn_vocab.stoi.get(rxn_s, 0)

        data.y_block = torch.tensor(y_block, dtype=torch.long)
        data.y_rxn = torch.tensor(y_rxn, dtype=torch.long)
        return data


