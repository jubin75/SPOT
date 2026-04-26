import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Optional

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphEncoder(nn.Module):
    """Simple MPNN-style encoder using stacked GCNConv + global mean pool.

    Produces one fixed-size embedding per input graph.
    """

    def __init__(self, node_feature_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        assert num_layers >= 1
        self.dropout = dropout
        layers: List[GCNConv] = []
        in_dim = node_feature_dim
        for _ in range(num_layers):
            layers.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.convs = nn.ModuleList(layers)

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        graph_embeddings = global_mean_pool(x, batch_idx)
        return graph_embeddings


class SynthPolicyNet(nn.Module):
    """
    合成策略网络 (Synthesis Policy Network, SynthPolicyNet)
    - 状态编码器: 将当前分子图 B_j 编码为 h(B_j)
    - 构建块编码器: 将全部构建块分子编码为向量库 E_block
    - 反应物选择头: 使用点积 h(B_j) · E_block^T 得到在构建块库上的分布
    - 反应类型选择头: 将 h(B_j) 与所选构建块向量拼接，预测反应模板分布
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        num_building_blocks: int,
        num_reaction_templates: int,
        num_gnn_layers: int = 3,
        dropout: float = 0.0,
        share_encoders: bool = False,
        use_l2_normalization: bool = True,
        initial_temperature: float = 0.07,
        enable_unconditional_rxn_head: bool = True,
    ):
        super().__init__()
        # Encoders
        self.state_encoder = GraphEncoder(node_feature_dim, hidden_dim, num_layers=num_gnn_layers, dropout=dropout)
        if share_encoders:
            self.block_encoder = self.state_encoder
        else:
            self.block_encoder = GraphEncoder(node_feature_dim, hidden_dim, num_layers=num_gnn_layers, dropout=dropout)

        # Reaction head takes concat(h_state, h_block)
        self.reaction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_reaction_templates),
        )

        self.hidden_dim = hidden_dim
        self.num_building_blocks = num_building_blocks
        self.num_reaction_templates = num_reaction_templates
        self.use_l2_normalization = use_l2_normalization
        # Temperature for cosine similarity scaling
        temp = max(1e-4, float(initial_temperature))
        self.temperature = nn.Parameter(torch.tensor(temp, dtype=torch.float32))

        # Optional unconditional reaction template head: P(rxn | state, protein)
        self.enable_unconditional_rxn_head = enable_unconditional_rxn_head
        if self.enable_unconditional_rxn_head:
            self.uncond_rxn_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_reaction_templates),
            )

        # Reaction-conditioned block selector: q(state, rxn) -> scores over blocks
        # Embed reaction template id then fuse with state to form a query
        self.rxn_embed = nn.Embedding(num_reaction_templates, hidden_dim)
        self.block_query_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

    @torch.no_grad()
    def encode_blocks(self, block_graphs: List, device: torch.device, batch_size: int = 512) -> torch.Tensor:
        """Encode all building block graphs into an embedding matrix E (num_blocks, hidden_dim).

        Note: weights will change during training; call this periodically (e.g., each epoch).
        """
        if len(block_graphs) == 0:
            return torch.zeros((0, self.hidden_dim), dtype=torch.float32, device=device)
        self.block_encoder.eval()
        loader = DataLoader(block_graphs, batch_size=batch_size, shuffle=False)
        embs: List[torch.Tensor] = []
        for b in loader:
            b = b.to(device)
            e = self.block_encoder(b)
            embs.append(e)
        embs_cat = torch.cat(embs, dim=0)
        if self.use_l2_normalization and embs_cat.numel() > 0:
            embs_cat = F.normalize(embs_cat, p=2, dim=-1)
        return embs_cat

    def compute_block_logits(self, h_state: torch.Tensor, block_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute logits over the building-block vocabulary via (scaled) cosine similarity or dot product.

        If L2 normalization is enabled, use cosine similarity scaled by 1/temperature. Otherwise, raw dot product.
        """
        if self.use_l2_normalization:
            # Normalize state features on the fly
            h_state = F.normalize(h_state, p=2, dim=-1)
            # block_embeddings expected normalized in encode_blocks; re-normalize defensively
            block_embeddings = F.normalize(block_embeddings, p=2, dim=-1)
            temp = torch.clamp(self.temperature, min=torch.tensor(1e-4, device=h_state.device, dtype=h_state.dtype))
            return (h_state @ block_embeddings.t()) / temp
        # Fallback to raw dot product
        return h_state @ block_embeddings.t()

    def compute_block_logits_given_rxn_h(
        self,
        h_state: torch.Tensor,
        block_embeddings: torch.Tensor,
        rxn_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Reaction-conditioned block logits given precomputed state features.

        Args:
            h_state: (B, D) state features (optionally protein-conditioned by caller)
            block_embeddings: (N_blocks, D)
            rxn_indices: (B,) LongTensor of reaction template ids
        Returns:
            block_logits: (B, N_blocks)
        """
        rxn_emb = self.rxn_embed(rxn_indices)  # (B, D)
        query = self.block_query_mlp(torch.cat([h_state, rxn_emb], dim=1))  # (B, D)
        if self.use_l2_normalization:
            query = F.normalize(query, p=2, dim=-1)
            block_embeddings = F.normalize(block_embeddings, p=2, dim=-1)
            temp = torch.clamp(self.temperature, min=torch.tensor(1e-4, device=query.device, dtype=query.dtype))
            return (query @ block_embeddings.t()) / temp
        return query @ block_embeddings.t()

    def compute_block_logits_given_rxn(
        self,
        state_batch,
        block_embeddings: torch.Tensor,
        rxn_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience wrapper to encode state and compute reaction-conditioned block logits."""
        h_state = self.state_encoder(state_batch)
        return self.compute_block_logits_given_rxn_h(h_state, block_embeddings, rxn_indices)

    def forward(
        self,
        state_batch,
        block_embeddings: torch.Tensor,
        block_indices_for_reaction: Optional[torch.Tensor] = None,
        return_uncond_rxn_logits: bool = False,
    ):
        """
        Args:
            state_batch: torch_geometric.data.Batch for current states (B graphs)
            block_embeddings: Tensor (num_blocks, hidden_dim) encoded by block_encoder
            block_indices_for_reaction: Optional LongTensor (B,) ground-truth block ids for reaction head

        Returns:
            block_logits: (B, num_blocks)
            rxn_logits: (B, num_reaction_templates) or None if block_indices_for_reaction is None
        """
        h_state = self.state_encoder(state_batch)  # (B, D)
        block_logits = self.compute_block_logits(h_state, block_embeddings)

        rxn_logits = None
        if block_indices_for_reaction is not None:
            # Gather selected/teacher-forced block embeddings
            selected_block_embs = block_embeddings.index_select(0, block_indices_for_reaction)
            rxn_input = torch.cat([h_state, selected_block_embs], dim=1)
            rxn_logits = self.reaction_head(rxn_input)
        if return_uncond_rxn_logits and getattr(self, "enable_unconditional_rxn_head", False):
            uncond_rxn_logits = self.uncond_rxn_head(h_state)
            return block_logits, rxn_logits, uncond_rxn_logits
        # Backward compatible return (2-tuple) when not explicitly requested
        return block_logits, rxn_logits
