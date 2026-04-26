#!/usr/bin/env python
from __future__ import annotations

import argparse
from typing import Iterable, List, Optional, Set

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# Silence RDKit parse errors (e.g., when encountering invalid strings)
RDLogger.DisableLog('rdApp.error')


def _split_on_plus_outside_brackets(s: str) -> List[str]:
    """Split on top-level ' + ' separators, not on '+' inside SMILES brackets.

    Heuristic: treat a '+' as a separator only when it has whitespace on both sides
    and we are not inside '[' ... ']'. This avoids splitting charged atoms like
    '[N+](C)(C)C'.
    """
    if not s:
        return []
    parts: List[str] = []
    buf: List[str] = []
    bracket_depth = 0
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if ch == '[':
            bracket_depth += 1
        elif ch == ']':
            bracket_depth = max(0, bracket_depth - 1)

        if ch == '+' and bracket_depth == 0:
            prev_c = s[i - 1] if i - 1 >= 0 else ''
            next_c = s[i + 1] if i + 1 < n else ''
            if prev_c.isspace() and next_c.isspace():
                token = ''.join(buf).strip()
                if token:
                    parts.append(token)
                buf = []
                i += 1
                continue

        buf.append(ch)
        i += 1

    token = ''.join(buf).strip()
    if token:
        parts.append(token)
    return parts


def _is_valid_smiles(smiles: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


def count_unique_reactants(
    csv_path: str,
    column: str = "action_building_block",
    *,
    validate: bool = True,
    ignore_empty: bool = True,
    chunksize: int = 500_000,
) -> int:
    unique: Set[str] = set()
    usecols = [column]

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        col = chunk[column]
        for raw in col:
            # Skip NA-like without converting to literal 'nan'
            if pd.isna(raw):
                if ignore_empty:
                    continue
                else:
                    unique.add("")
                    continue

            token_str = str(raw).strip()
            if token_str.lower() in {"nan", "na", "n/a", ""}:
                if ignore_empty:
                    continue
                else:
                    unique.add("")
                    continue
            if not token_str:
                if ignore_empty:
                    continue
                else:
                    unique.add("")
                    continue

            # Split possible multi-token fields (rare if skip-start-steps used)
            tokens = _split_on_plus_outside_brackets(token_str)
            if not tokens:
                tokens = [token_str]
            for t in tokens:
                t = t.strip()
                if not t and ignore_empty:
                    continue
                # Skip NA-like tokens after split
                tl = t.lower()
                if tl in {"nan", "na", "n/a"}:
                    if ignore_empty:
                        continue
                    t = ""
                if validate and not _is_valid_smiles(t):
                    continue
                unique.add(t)

    return len(unique)


def main() -> None:
    p = argparse.ArgumentParser(description="Count unique reactant building blocks in forward_trajectories.csv")
    p.add_argument("--input", default="data/forward_trajectories.csv", help="Path to forward trajectories CSV")
    p.add_argument("--column", default="action_building_block", help="Column containing building block SMILES")
    p.add_argument("--no-validate", action="store_true", help="Do not validate SMILES with RDKit")
    p.add_argument("--include-empty", action="store_true", help="Include empty tokens in the count")
    p.add_argument("--chunksize", type=int, default=500000, help="Rows per chunk when streaming CSV")
    args = p.parse_args()

    n = count_unique_reactants(
        args.input,
        column=args.column,
        validate=(not args.no_validate),
        ignore_empty=(not args.include_empty),
        chunksize=int(args.chunksize),
    )
    print({"unique_reactants": n, "input": args.input, "column": args.column, "validated": (not args.no_validate)})


if __name__ == "__main__":
    main()


