from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def batched_topk_indices(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    topk: int,
    *,
    normalize: bool = True,
    corpus_chunk_size: int = 4096,
) -> torch.Tensor:
    """Return per-query top-k indices into the corpus by cosine similarity or dot-product.

    Args:
        query_embeddings: (B, D)
        corpus_embeddings: (N, D)
        topk: number of nearest items per query to return
        normalize: if True, L2-normalize queries and corpus before scoring
        corpus_chunk_size: process corpus in chunks to limit memory

    Returns:
        indices: (B, K) LongTensor of top-k indices per query
    """
    assert query_embeddings.dim() == 2 and corpus_embeddings.dim() == 2
    assert query_embeddings.size(1) == corpus_embeddings.size(1)
    device = query_embeddings.device
    dtype = query_embeddings.dtype

    B, D = query_embeddings.shape
    N = corpus_embeddings.shape[0]
    K = min(max(1, int(topk)), int(N))

    if normalize:
        q = F.normalize(query_embeddings, p=2, dim=-1)
        c = F.normalize(corpus_embeddings, p=2, dim=-1)
    else:
        q = query_embeddings
        c = corpus_embeddings

    # Initialize running top-K buffers
    running_scores = torch.full((B, K), fill_value=-1e9, device=device, dtype=dtype)
    running_indices = torch.full((B, K), fill_value=-1, device=device, dtype=torch.long)

    # Process corpus in chunks
    for start in range(0, N, corpus_chunk_size):
        end = min(start + corpus_chunk_size, N)
        c_chunk = c[start:end]  # (C, D)
        # (B, D) @ (D, C) = (B, C)
        scores_chunk = q @ c_chunk.t()

        # Merge chunk top-K with running top-K
        # Concatenate scores along candidate dimension: (B, K + C)
        combined_scores = torch.cat([running_scores, scores_chunk], dim=1)
        # Build corresponding indices
        # running indices are already absolute; chunk indices need offset
        chunk_indices = torch.arange(start, end, device=device, dtype=torch.long)
        chunk_indices = chunk_indices.unsqueeze(0).expand(B, -1)  # (B, C)
        combined_indices = torch.cat([running_indices, chunk_indices], dim=1)

        # Take top-K over combined
        top_scores, top_pos = torch.topk(combined_scores, k=K, dim=1)
        top_indices = combined_indices.gather(1, top_pos)

        running_scores = top_scores
        running_indices = top_indices

    return running_indices


def build_union_of_candidates(
    per_query_topk: torch.Tensor,
    required_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the union set of candidate indices across the batch.

    Args:
        per_query_topk: (B, K) LongTensor
        required_indices: Optional (B,) LongTensor of indices that must be included

    Returns:
        union_indices: (M,) LongTensor of unique indices (order stable by first occurrence)
    """
    B, K = per_query_topk.shape
    flat = per_query_topk.reshape(-1)
    if required_indices is not None:
        flat = torch.cat([flat, required_indices.view(-1)], dim=0)

    # Stable unique: use seen mask
    # Note: for large vocab, torch.unique preserves order if return_inverse=False (PyTorch 2.x behavior is not order-stable).
    # Implement manual stable unique via Python set on CPU for robustness, but keep tensors on device by gathering later.
    flat_cpu = flat.detach().to("cpu").tolist()
    seen = set()
    union_list = []
    for idx in flat_cpu:
        if idx not in seen:
            seen.add(idx)
            union_list.append(idx)
    return torch.tensor(union_list, dtype=torch.long, device=per_query_topk.device)


