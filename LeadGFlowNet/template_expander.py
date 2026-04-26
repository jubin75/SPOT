from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import random

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdkit import RDLogger

# Suppress RDKit warnings (e.g., unmapped atoms notices) without changing behavior
RDLogger.DisableLog('rdApp.warning')
try:
    RDLogger.DisableLog('rdApp.error')
except Exception:
    pass


def _count_left_components(rxn_smarts: str) -> int:
    try:
        left = rxn_smarts.split(">>")[0]
        return max(0, len([t for t in left.split(".") if t.strip()]))
    except Exception:
        return 0


@dataclass
class CompiledTemplate:
    smarts: str
    label: str
    left_n: int
    rxn: rdChemReactions.ChemicalReaction


@dataclass
class TemplateStep:
    """One application of a template, with enough metadata for visualization."""

    product_smiles: str
    block_smiles: Optional[str]
    template_label: str
    template_smarts: str
    left_n: int


class TemplateLibrary:
    def __init__(self, compiled: List[CompiledTemplate]):
        self.compiled = compiled

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        *,
        smarts_col: str = "updated_reaction",
        label_col: str = "mechanistic_label",
        max_rows: int = 5000,
        seed: int = 42,
    ) -> "TemplateLibrary":
        # Load CSV/XLSX with flexible column detection for SMARTS
        path_lower = str(csv_path).lower()
        df_all = None
        last_exc: Optional[Exception] = None
        # Prefer Excel reader for .xlsx/.xls
        if path_lower.endswith((".xlsx", ".xls")):
            try:
                df_all = pd.read_excel(csv_path)
            except Exception as e:  # fall back to CSV reader if Excel engine is unavailable
                last_exc = e
        if df_all is None:
            try:
                df_all = pd.read_csv(csv_path)
            except Exception as e:
                last_exc = e
        if df_all is None:
            # Re-raise the last exception to be handled by caller
            raise last_exc if last_exc is not None else RuntimeError(f"Failed to load template file: {csv_path}")
        # Prefer provided smarts_col, else auto-detect common names
        if smarts_col not in df_all.columns:
            for alt in ["reaction_smarts", "smarts", "rxn_smarts", "updated_reaction"]:
                if alt in df_all.columns:
                    smarts_col = alt
                    break
        # Fallback: scan for a column containing SMIRKS-like patterns with '>>'
        if smarts_col not in df_all.columns:
            candidate = None
            for c in df_all.columns:
                try:
                    series = df_all[c].astype(str)
                    hit = series.str.contains(">>", regex=False, na=False).sum()
                    if hit >= max(5, int(0.05 * len(series))):
                        candidate = c
                        break
                except Exception:
                    continue
            if candidate is not None:
                smarts_col = candidate
        # Detect an optional label column if the provided one does not exist
        if label_col not in df_all.columns:
            for lc in ["mechanistic_label", "label", "name", "class", "type", "reaction_name"]:
                if lc in df_all.columns:
                    label_col = lc
                    break
        usecols = [smarts_col]
        if label_col in df_all.columns:
            usecols.append(label_col)
        df = df_all[usecols].copy()
        df = df.dropna(subset=[smarts_col]).drop_duplicates(subset=[smarts_col]).reset_index(drop=True)
        if max_rows and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=seed)
        compiled: List[CompiledTemplate] = []
        for _, r in df.iterrows():
            s = str(r[smarts_col]).strip()
            if ">>" not in s:
                continue
            try:
                # Parse as SMARTS/SMIRKS so generic reactant patterns can match substructures
                rxn = rdChemReactions.ReactionFromSmarts(s, useSmiles=False)
                if rxn is None:
                    continue
            except Exception:
                continue
            left_n = _count_left_components(s)
            label = str(r.get(label_col, "")) if label_col in r else ""
            compiled.append(CompiledTemplate(smarts=s, label=label, left_n=left_n, rxn=rxn))
        # Keep only 1- or 2-reactant templates (compatible with state[,block])
        compiled = [t for t in compiled if t.left_n in (1, 2)]
        return cls(compiled)

    def propose_steps(
        self,
        state_smiles: str,
        block_smiles: Optional[str] = None,
        *,
        try_limit: int = 64,
        max_products_per_template: int = 2,
        seed: Optional[int] = None,
    ) -> List[TemplateStep]:
        if seed is not None:
            random.seed(int(seed))
        steps: List[TemplateStep] = []
        state_mol = Chem.MolFromSmiles(state_smiles) if state_smiles else None
        block_mol = Chem.MolFromSmiles(block_smiles) if block_smiles else None
        if state_mol is None and block_mol is None:
            return []
        # Shuffle templates for variety
        idxs = list(range(len(self.compiled)))
        random.shuffle(idxs)
        tries = 0
        for i in idxs:
            if tries >= max(1, int(try_limit)):
                break
            tpl = self.compiled[i]
            # Decide reactant tuple strictly by left_n
            reactant_sets: List[Tuple[Chem.Mol, ...]] = []
            if tpl.left_n == 1:
                if state_mol is not None:
                    reactant_sets.append((state_mol,))
            elif tpl.left_n == 2:
                if state_mol is not None and block_mol is not None:
                    reactant_sets.append((state_mol, block_mol))
                    reactant_sets.append((block_mol, state_mol))
            # Skip 3+ reactant templates
            if not reactant_sets:
                continue
            tries += 1
            for reactants in reactant_sets:
                try:
                    outs = tpl.rxn.RunReactants(tuple(reactants))
                except Exception:
                    continue
                # outs: list of tuples of product mols sets; take first product of each outcome
                for outcome in outs[: max_products_per_template]:
                    if not outcome:
                        continue
                    # Choose the largest product as main
                    try:
                        ms = sorted(outcome, key=lambda m: m.GetNumAtoms(), reverse=True)
                    except Exception:
                        ms = list(outcome)
                    main = ms[0]
                    try:
                        smiles = Chem.MolToSmiles(main)
                        if smiles:
                            steps.append(
                                TemplateStep(
                                    product_smiles=smiles,
                                    block_smiles=block_smiles if tpl.left_n == 2 else None,
                                    template_label=tpl.label,
                                    template_smarts=tpl.smarts,
                                    left_n=tpl.left_n,
                                )
                            )
                    except Exception:
                        continue
        # Deduplicate by product SMILES, keep first metadata
        seen = set()
        uniq: List[TemplateStep] = []
        for st in steps:
            s = st.product_smiles
            if s not in seen:
                seen.add(s)
                uniq.append(st)
        return uniq

    def propose_products(
        self,
        state_smiles: str,
        block_smiles: Optional[str] = None,
        *,
        try_limit: int = 64,
        max_products_per_template: int = 2,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Backward-compatible wrapper that returns only product SMILES."""
        steps = self.propose_steps(
            state_smiles=state_smiles,
            block_smiles=block_smiles,
            try_limit=try_limit,
            max_products_per_template=max_products_per_template,
            seed=seed,
        )
        return [st.product_smiles for st in steps]

    def propose_steps_with_pool(
        self,
        state_smiles: str,
        block_pool: List[str],
        *,
        try_limit_templates: int = 64,
        sample_blocks: int = 64,
        max_products_per_template: int = 2,
        seed: Optional[int] = None,
    ) -> List[TemplateStep]:
        if seed is not None:
            random.seed(int(seed))
        steps: List[TemplateStep] = []
        state_mol = Chem.MolFromSmiles(state_smiles) if state_smiles else None
        if state_mol is None:
            return []
        # Shuffle templates for variety
        idxs = list(range(len(self.compiled)))
        random.shuffle(idxs)
        ttries = 0
        for i in idxs:
            if ttries >= max(1, int(try_limit_templates)):
                break
            ttries += 1
            tpl = self.compiled[i]
            if tpl.left_n == 1:
                # Direct attempt without pre-filter to avoid false negatives in pattern checks
                try:
                    outs = tpl.rxn.RunReactants((state_mol,))
                except Exception:
                    outs = []
                for outcome in outs[: max_products_per_template]:
                    if not outcome:
                        continue
                    try:
                        ms = sorted(outcome, key=lambda m: m.GetNumAtoms(), reverse=True)
                    except Exception:
                        ms = list(outcome)
                    main = ms[0]
                    try:
                        smiles = Chem.MolToSmiles(main)
                        if smiles:
                            steps.append(
                                TemplateStep(
                                    product_smiles=smiles,
                                    block_smiles=None,
                                    template_label=tpl.label,
                                    template_smarts=tpl.smarts,
                                    left_n=tpl.left_n,
                                )
                            )
                    except Exception:
                        continue
                # Early return if any product found
                if products:
                    break
                continue

            # Two-reactant template: try both (state, block) and (block, state) orderings
            if tpl.left_n == 2:
                # Sample a subset of blocks for attempts
                if not block_pool:
                    continue
                # Draw without replacement up to sample_blocks
                if len(block_pool) <= sample_blocks:
                    blocks_sampled = list(block_pool)
                    random.shuffle(blocks_sampled)
                else:
                    blocks_sampled = random.sample(block_pool, k=int(sample_blocks))

                for bs in blocks_sampled:
                    bm = Chem.MolFromSmiles(bs)
                    if bm is None:
                        continue
                    # Try both orderings to avoid side-specific pre-filters
                    pairs = [(state_mol, bm), (bm, state_mol)]
                    ok = False
                    for reactants in pairs:
                        try:
                            outs = tpl.rxn.RunReactants(tuple(reactants))
                        except Exception:
                            outs = []
                        for outcome in outs[: max_products_per_template]:
                            if not outcome:
                                continue
                            try:
                                ms = sorted(outcome, key=lambda m: m.GetNumAtoms(), reverse=True)
                            except Exception:
                                ms = list(outcome)
                            main = ms[0]
                            try:
                                smiles = Chem.MolToSmiles(main)
                                if smiles:
                                    steps.append(
                                        TemplateStep(
                                            product_smiles=smiles,
                                            block_smiles=bs,
                                            template_label=tpl.label,
                                            template_smarts=tpl.smarts,
                                            left_n=tpl.left_n,
                                        )
                                    )
                                    ok = True
                            except Exception:
                                continue
                        if ok:
                            break
                    if ok:
                        break
                if products:
                    break

        # Deduplicate by product SMILES, keep first metadata
        seen = set()
        uniq: List[TemplateStep] = []
        for st in steps:
            s = st.product_smiles
            if s not in seen:
                seen.add(s)
                uniq.append(st)
        return uniq

    def propose_products_with_pool(
        self,
        state_smiles: str,
        block_pool: List[str],
        *,
        try_limit_templates: int = 64,
        sample_blocks: int = 64,
        max_products_per_template: int = 2,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Backward-compatible wrapper that returns only product SMILES."""
        steps = self.propose_steps_with_pool(
            state_smiles=state_smiles,
            block_pool=block_pool,
            try_limit_templates=try_limit_templates,
            sample_blocks=sample_blocks,
            max_products_per_template=max_products_per_template,
            seed=seed,
        )
        return [st.product_smiles for st in steps]


