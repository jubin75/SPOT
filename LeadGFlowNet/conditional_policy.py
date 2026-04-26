# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from SynthPolicyNet.models import SynthPolicyNet


class ConditionalSynthPolicy(nn.Module):
    """Condition SynthPolicyNet on a protein embedding h(A).

    FiLM-style conditioning: learn per-dimension scale and shift from protein embedding.
    - For block head: h_state' = gamma(h_prot) ⊙ h_state + beta(h_prot)
    - For rxn head: concat(h_state', selected_block_embs)
    """

    def __init__(self, base: SynthPolicyNet, protein_dim: int):
        super().__init__()
        self.base = base
        hidden = base.hidden_dim
        # FiLM generators
        self.gamma_mlp = nn.Sequential(
            nn.Linear(protein_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),  # range [-1,1]
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(protein_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        # STOP action head (produces a single logit for STOP)
        self.stop_head = nn.Linear(hidden, 1)
        # Optional action-source embedding (0: dataset/internal, 1: template, 2: free-connect)
        # Added to the FiLM-conditioned state representation when provided.
        self.source_embed = nn.Embedding(3, hidden)

    def compute_h_state_block(self, state_batch, protein_emb: torch.Tensor) -> torch.Tensor:
        """Encode state and fuse protein via FiLM for block selection/reaction heads."""
        h_state = self.base.state_encoder(state_batch)
        gamma = self.gamma_mlp(protein_emb)
        beta = self.beta_mlp(protein_emb)
        h_state_block = gamma * h_state + beta
        return h_state_block

    def compute_h_state_block_with_source(self, state_batch, protein_emb: torch.Tensor, source_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Like compute_h_state_block, but optionally add an action-source embedding.

        Args:
            source_ids: (B,) LongTensor with values in {0,1,2} indicating action provenance.
        """
        h = self.compute_h_state_block(state_batch, protein_emb)
        if source_ids is not None:
            # Ensure shape (B,) long
            if source_ids.dtype != torch.long:
                source_ids = source_ids.long()
            h = h + self.source_embed(source_ids)
        return h

    def forward(
        self,
        state_batch,
        block_embeddings: torch.Tensor,
        protein_emb: torch.Tensor,
        block_indices_for_reaction: Optional[torch.Tensor] = None,
    ):
        # Encode + fuse (FiLM)
        h_state_block = self.compute_h_state_block(state_batch, protein_emb)
        block_logits = self.base.compute_block_logits(h_state_block, block_embeddings)

        rxn_logits = None
        if block_indices_for_reaction is not None:
            selected_block_embs = block_embeddings.index_select(0, block_indices_for_reaction)
            rxn_input = torch.cat([h_state_block, selected_block_embs], dim=1)
            rxn_logits = self.base.reaction_head(rxn_input)
        return block_logits, rxn_logits

    def logits_with_source(
        self,
        state_batch,
        block_embeddings: torch.Tensor,
        protein_emb: torch.Tensor,
        *,
        block_indices_for_reaction: Optional[torch.Tensor] = None,
        source_ids: Optional[torch.Tensor] = None,
    ):
        """Compute block/rxn logits while conditioning on optional action-source ids.

        This is useful for training/evaluating a learned backward policy P_B(· | child, source).
        """
        h_state_block = self.compute_h_state_block_with_source(state_batch, protein_emb, source_ids)
        block_logits = self.base.compute_block_logits(h_state_block, block_embeddings)
        rxn_logits = None
        if block_indices_for_reaction is not None:
            selected_block_embs = block_embeddings.index_select(0, block_indices_for_reaction)
            rxn_input = torch.cat([h_state_block, selected_block_embs], dim=1)
            rxn_logits = self.base.reaction_head(rxn_input)
        return block_logits, rxn_logits

    def rxn_first(
        self,
        state_batch,
        block_embeddings: torch.Tensor,
        protein_emb: torch.Tensor,
        rxn_indices_for_blocks: Optional[torch.Tensor] = None,
    ):
        """Reaction-first path:
        1) Predict rxn logits unconditionally from protein-conditioned state.
        2) Score blocks conditioned on an rxn (teacher or predicted) via base.compute_block_logits_given_rxn_h.
        """
        h_state_block = self.compute_h_state_block(state_batch, protein_emb)
        # Unconditional rxn on protein-conditioned state
        uncond_rxn_logits = self.base.uncond_rxn_head(h_state_block)
        block_logits = None
        if rxn_indices_for_blocks is not None:
            block_logits = self.base.compute_block_logits_given_rxn_h(h_state_block, block_embeddings, rxn_indices_for_blocks)
        return uncond_rxn_logits, block_logits

    def rxn_logits_with_stop(self, h_state_block: torch.Tensor, selected_block_embs: Optional[torch.Tensor]) -> torch.Tensor:
        """Return reaction logits augmented with a STOP logit at the last column.

        Output shape: (B, num_reaction_templates + 1). The last index corresponds to STOP.
        """
        if selected_block_embs is None:
            selected_block_embs = torch.zeros_like(h_state_block)
        rxn_input = torch.cat([h_state_block, selected_block_embs], dim=1)
        rxn_logits = self.base.reaction_head(rxn_input)
        stop_logit = self.stop_head(h_state_block)  # (B, 1)
        return torch.cat([rxn_logits, stop_logit], dim=1)


