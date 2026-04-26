import argparse
import os
from typing import Dict

import sys

import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pandas as pd

from SynthPolicyNet.datasets import ForwardTrajectoryDataset, Vocab
from SynthPolicyNet.models import SynthPolicyNet


def build_forward_dataset(
    input_csv: str,
    forward_csv: str,
    skip_start_steps: bool = True,
    rebuild: bool = False,
    max_block_mw: float | None = 200.0,
    max_state_mw: float | None = None,
    max_ligand_mw: float | None = None,
) -> pd.DataFrame:
    """Ensure forward trajectories CSV exists; build from retrosynthesis CSV if needed."""
    if os.path.exists(forward_csv) and not rebuild:
        # Read without converting empty strings to NaN
        return pd.read_csv(forward_csv, keep_default_na=False)
    # Fallback: try to build via the provided converter script
    from scripts.forward_trajectories import load_dataset, convert, assign_forward_order

    df = load_dataset(input_csv)
    out = convert(
        df,
        skip_start_steps=skip_start_steps,
        max_block_mw=max_block_mw,
        max_state_mw=max_state_mw,
        max_ligand_mw=max_ligand_mw,
    )
    out = assign_forward_order(out)
    os.makedirs(os.path.dirname(forward_csv), exist_ok=True)
    out.to_csv(forward_csv, index=False)
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train SynthPolicyNet via Behavior Cloning")
    p.add_argument("--input", default="data/reaction_paths_all_routes.csv", help="Retrosynthesis CSV path")
    p.add_argument("--forward", default="data/forward_trajectories.csv", help="Forward trajectories CSV path")
    p.add_argument("--rebuild-forward", action="store_true", help="Rebuild forward trajectories even if CSV exists")
    p.add_argument("--max-block-mw", type=float, default=200.0, help="Maximum molecular weight for building block reactants (Da)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num-gnn-layers", type=int, default=3)
    p.add_argument("--share-encoders", action="store_true")
    p.add_argument("--use-cpu", action="store_true")
    p.add_argument("--save", default="checkpoints/synth_policy_net.pt")
    p.add_argument("--max-samples", type=int, default=0, help="If >0, train on a random subset of this many samples")

    # Loss weighting and regularization
    p.add_argument("--w-block", type=float, default=1.0, help="Weight for block classification loss")
    p.add_argument("--w-rxn", type=float, default=1.0, help="Weight for reaction classification loss")
    p.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for CE losses")

    # Optional warmup phase where reaction head is frozen
    p.add_argument("--freeze-rxn-epochs", type=int, default=0, help="Freeze reaction head for first N epochs")

    # Scheduler
    p.add_argument("--scheduler", choices=["none", "cosine"], default="none", help="LR scheduler type")
    p.add_argument("--eta-min", type=float, default=1e-5, help="Min LR for cosine scheduler")

    # Reaction-first option
    p.add_argument("--rxn-first", action="store_true", help="Train in reaction-first mode: predict rxn unconditionally, then block conditioned on rxn")
    p.add_argument("--rxn-sched-prob", type=float, default=0.0, help="Probability to condition block on predicted rxn instead of teacher rxn (scheduled sampling)")
    return p


def train() -> None:
    args = build_argparser().parse_args()

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # 1) Prepare forward data
    fwd_df = build_forward_dataset(
        args.input,
        args.forward,
        skip_start_steps=True,
        rebuild=args.rebuild_forward,
        max_block_mw=args.max_block_mw,
    )
    
    # Debug: Check for NaN values in SMILES columns
    print("=== DEBUG: Checking for NaN values in forward trajectories ===")
    print(f"DataFrame shape: {fwd_df.shape}")
    print(f"DataFrame columns: {list(fwd_df.columns)}")
    
    # Check all columns for any type of NaN values
    for col in fwd_df.columns:
        nan_count = fwd_df[col].isna().sum()
        if nan_count > 0:
            print(f"Column '{col}' has {nan_count} NaN values")
            # Show first few NaN examples
            nan_examples = fwd_df[fwd_df[col].isna()].head(3)
            print(f"First NaN examples in {col}:")
            print(nan_examples)
    
    # Check for 'nan' string values (which cause SMILES parse errors)
    for col in fwd_df.columns:
        if fwd_df[col].dtype == 'object':  # String columns
            nan_string_count = (fwd_df[col] == 'nan').sum()
            if nan_string_count > 0:
                print(f"Column '{col}' has {nan_string_count} 'nan' string values")
                # Show first few 'nan' string examples
                nan_string_examples = fwd_df[fwd_df[col] == 'nan'].head(3)
                print(f"First 'nan' string examples in {col}:")
                print(nan_string_examples)
            
            # Also check for other problematic values
            empty_count = (fwd_df[col] == '').sum()
            if empty_count > 0:
                print(f"Column '{col}' has {empty_count} empty string values")
            
            # Check for any string that might cause SMILES parsing issues
            problematic_values = fwd_df[col].value_counts()
            if len(problematic_values) < 20:  # Only show if not too many unique values
                print(f"Column '{col}' unique values: {problematic_values.to_dict()}")
    
    # NEW: Save problematic data to file for inspection
    problematic_rows = pd.DataFrame()
    for col in fwd_df.columns:
        if fwd_df[col].dtype == 'object':
            is_problematic = (fwd_df[col] == 'nan') | (fwd_df[col] == '') | (fwd_df[col].isna())
            if is_problematic.sum() > 0:
                problematic_rows = pd.concat([problematic_rows, fwd_df[is_problematic]], axis=0)
    
    # NEW: Check for potentially invalid SMILES strings
    try:
        from rdkit import Chem
        smiles_columns = [col for col in fwd_df.columns if 'smiles' in col.lower() or 'molecule' in col.lower()]
        for col in smiles_columns:
            if fwd_df[col].dtype == 'object':
                invalid_smiles_rows = []
                for idx, smiles in fwd_df[col].items():
                    if smiles and smiles != 'nan' and smiles != '' and not pd.isna(smiles):
                        try:
                            mol = Chem.MolFromSmiles(smiles)
                            if mol is None:
                                invalid_smiles_rows.append(idx)
                        except Exception as e:
                            invalid_smiles_rows.append(idx)
                if invalid_smiles_rows:
                    print(f"Column '{col}' has {len(invalid_smiles_rows)} rows with potentially invalid SMILES")
                    invalid_df = fwd_df.loc[invalid_smiles_rows]
                    problematic_rows = pd.concat([problematic_rows, invalid_df], axis=0)
                    # Show first few invalid SMILES
                    invalid_examples = invalid_df.head(3)
                    print(f"First invalid SMILES examples in {col}:")
                    for _, row in invalid_examples.iterrows():
                        print(f"  Row {row.name}: {row[col]}")
    except ImportError:
        print("RDKit not available, skipping SMILES validation")
    
    if not problematic_rows.empty:
        problematic_rows = problematic_rows.drop_duplicates()
        print(f"Found {len(problematic_rows)} problematic rows with 'nan', empty, or invalid SMILES values")
        problematic_file = 'debug_problematic_rows.csv'
        problematic_rows.to_csv(problematic_file, index=False)
        print(f"Saved problematic rows to {problematic_file} for inspection")
    else:
        print("No problematic rows found with 'nan', empty, or invalid SMILES values")
    
    # NEW: Clean the data before passing to dataset
    print("Cleaning data by replacing 'nan' strings with empty string")
    for col in fwd_df.columns:
        if fwd_df[col].dtype == 'object':
            fwd_df[col] = fwd_df[col].replace('nan', '')
            fwd_df[col] = fwd_df[col].fillna('')
    print("Data cleaning complete")
    
    # NEW: Filter out rows with NaN values in critical columns
    critical_columns = ['action_building_block', 'state_smiles']
    original_row_count = len(fwd_df)
    for col in critical_columns:
        if col in fwd_df.columns:
            fwd_df = fwd_df[fwd_df[col].notna()]
    filtered_row_count = len(fwd_df)
    print(f"Filtered data: {original_row_count} rows -> {filtered_row_count} rows (removed {original_row_count - filtered_row_count} rows with NaN in critical columns)")
    
    print("=== End DEBUG ===")
    
    # Dataset + vocabs
    print("=== DEBUG: Creating ForwardTrajectoryDataset ===")
    try:
        dataset = ForwardTrajectoryDataset(
            fwd_df,
            block_vocab=None,
            rxn_vocab=None,
            use_only_forward_chain=True,
            skip_start_states=True,
        )
        print("Dataset created successfully!")
    except Exception as e:
        print(f"ERROR creating dataset: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to identify problematic rows
        print("\n=== DEBUG: Checking data types and sample values ===")
        for col in fwd_df.columns:
            print(f"Column '{col}': dtype={fwd_df[col].dtype}, sample_values={fwd_df[col].head(3).tolist()}")
        
        # Check for any 'nan' strings in all columns
        for col in fwd_df.columns:
            if fwd_df[col].dtype == 'object':
                nan_string_count = (fwd_df[col] == 'nan').sum()
                if nan_string_count > 0:
                    print(f"Column '{col}' has {nan_string_count} 'nan' string values")
                    nan_rows = fwd_df[fwd_df[col] == 'nan'].head(3)
                    print(f"Sample rows with 'nan' in {col}:")
                    print(nan_rows)
        raise e
    if args.max_samples and args.max_samples > 0 and len(dataset) > args.max_samples:
        import random
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[: args.max_samples]
        from torch.utils.data import Subset
        dataset_to_train = Subset(dataset, indices)
    else:
        dataset_to_train = dataset

    loader = DataLoader(dataset_to_train, batch_size=args.batch_size, shuffle=True)

    print(
        {
            "samples": len(dataset),
            "unique_blocks": len(dataset.block_vocab.itos),
            "unique_rxns": len(dataset.rxn_vocab.itos),
        }
    )

    # 2) Build block embedding cache (graphs)
    print("=== DEBUG: Building block embedding cache ===")
    print(f"Total block graphs: {len(dataset.block_graphs)}")
    
    # Check for None graphs and print details
    none_graphs = [i for i, g in enumerate(dataset.block_graphs) if g is None]
    if none_graphs:
        print(f"Found {len(none_graphs)} None graphs at indices: {none_graphs[:10]}...")
        # Show corresponding SMILES for None graphs
        for idx in none_graphs[:5]:
            if hasattr(dataset, 'block_smiles') and idx < len(dataset.block_smiles):
                print(f"  Index {idx}: SMILES = {dataset.block_smiles[idx]}")
    
    # Filter out None graphs with simple single-node fallback handled in model.encode_blocks
    valid_block_graphs = [g if g is not None else Data(x=torch.zeros((1, dataset.node_feature_dim)), edge_index=torch.zeros((2, 0), dtype=torch.long)) for g in dataset.block_graphs]
    print(f"Valid block graphs: {len(valid_block_graphs)}")
    print("=== End DEBUG ===")

    # 3) Model
    model = SynthPolicyNet(
        node_feature_dim=dataset.node_feature_dim,
        hidden_dim=args.hidden_dim,
        num_building_blocks=len(dataset.block_vocab.itos),
        num_reaction_templates=len(dataset.rxn_vocab.itos),
        num_gnn_layers=args.num_gnn_layers,
        dropout=args.dropout,
        share_encoders=args.share_encoders,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    ce_block = torch.nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    ce_rxn = torch.nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))

    # Scheduler
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(args.epochs), eta_min=float(args.eta_min)
        )

    # Optional: freeze reaction head for initial epochs
    rxn_frozen = False
    if int(args.freeze_rxn_epochs) > 0:
        if hasattr(model, "reaction_head"):
            for p in model.reaction_head.parameters():
                p.requires_grad = False
        if getattr(model, "enable_unconditional_rxn_head", False) and hasattr(model, "uncond_rxn_head"):
            for p in model.uncond_rxn_head.parameters():
                p.requires_grad = False
        rxn_frozen = True

    print(
        "Training config:",
        {
            "w_block": float(args.w_block),
            "w_rxn": float(args.w_rxn),
            "freeze_rxn_epochs": int(args.freeze_rxn_epochs),
            "scheduler": str(args.scheduler),
            "eta_min": float(args.eta_min),
            "label_smoothing": float(args.label_smoothing),
            "lr": float(args.lr),
        },
    )

    # 4) Training loop
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    best_loss = float("inf")
    
    # Track loss history for monitoring
    loss_history = {'block': [], 'rxn': [], 'total': []}
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Unfreeze reaction head when warmup finishes
        if rxn_frozen and epoch > int(args.freeze_rxn_epochs):
            if hasattr(model, "reaction_head"):
                for p in model.reaction_head.parameters():
                    p.requires_grad = True
            if getattr(model, "enable_unconditional_rxn_head", False) and hasattr(model, "uncond_rxn_head"):
                for p in model.uncond_rxn_head.parameters():
                    p.requires_grad = True
            rxn_frozen = False

        effective_w_rxn = float(args.w_rxn) if epoch > int(args.freeze_rxn_epochs) else 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Start Epoch {epoch:03d} | lr={current_lr:.2e} w_block={float(args.w_block):.2f} w_rxn={effective_w_rxn:.2f} rxn_first={bool(args.rxn_first)}")
        # Refresh block embeddings each epoch
        try:
            block_embs = model.encode_blocks(valid_block_graphs, device=device)
        except Exception as e:
            print(f"ERROR in encode_blocks at epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            raise e

        total_block_loss = 0.0
        total_rxn_loss = 0.0
        count = 0

        for batch_idx, batch in enumerate(loader):
            try:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)

                if args.rxn_first:
                    # 1) Unconditional rxn logits: P(rxn | state)
                    h_state = model.state_encoder(batch)
                    uncond_rxn_logits = model.uncond_rxn_head(h_state)
                    # Unify name for downstream debug checks
                    rxn_logits = uncond_rxn_logits
                    loss_rxn = ce_rxn(uncond_rxn_logits, batch.y_rxn)

                    # 2) Condition block on rxn (teacher-forced by default)
                    with torch.no_grad():
                        rxn_pred = torch.argmax(uncond_rxn_logits, dim=1)
                    if float(args.rxn_sched_prob) > 0.0:
                        # Scheduled sampling: pick predicted with prob p else teacher
                        p = float(args.rxn_sched_prob)
                        rand_mask = torch.rand_like(batch.y_rxn.float()) < p
                        rxn_used = torch.where(rand_mask, rxn_pred, batch.y_rxn)
                    else:
                        rxn_used = batch.y_rxn
                    block_logits = model.compute_block_logits_given_rxn_h(h_state, block_embs, rxn_used)
                    loss_block = ce_block(block_logits, batch.y_block)
                    loss = float(args.w_block) * loss_block + effective_w_rxn * loss_rxn
                else:
                    block_logits, rxn_logits = model(
                        state_batch=batch,
                        block_embeddings=block_embs,
                        block_indices_for_reaction=batch.y_block,
                    )

                    loss_block = ce_block(block_logits, batch.y_block)
                    loss_rxn = ce_rxn(rxn_logits, batch.y_rxn)
                    loss = float(args.w_block) * loss_block + effective_w_rxn * loss_rxn
                loss.backward()
                optimizer.step()

                bs = batch.num_graphs
                total_block_loss += loss_block.item() * bs
                total_rxn_loss += loss_rxn.item() * bs
                count += bs
                
                # Debug: Print first few batches to check for 'nan' values
                if epoch == 1 and batch_idx < 3:
                    print(f"DEBUG Batch {batch_idx}: block_loss={loss_block.item():.4f}, rxn_loss={loss_rxn.item():.4f}")
                    # Check for NaN in logits
                    if torch.isnan(block_logits).any():
                        print(f"  WARNING: NaN found in block_logits at batch {batch_idx}")
                    if rxn_logits is not None and torch.isnan(rxn_logits).any():
                        print(f"  WARNING: NaN found in rxn_logits at batch {batch_idx}")
                        
            except Exception as e:
                print(f"ERROR processing batch {batch_idx} at epoch {epoch}: {e}")
                batch_device = getattr(getattr(batch, 'x', None), 'device', device)
                print(f"Batch info: num_graphs={batch.num_graphs}, device={batch_device}")
                import traceback
                traceback.print_exc()
                raise e

        avg_block = total_block_loss / max(1, count)
        avg_rxn = total_rxn_loss / max(1, count)
        avg_total = float(args.w_block) * avg_block + effective_w_rxn * avg_rxn
        
        # Store loss history
        loss_history['block'].append(avg_block)
        loss_history['rxn'].append(avg_rxn)
        loss_history['total'].append(avg_total)
        
        # Print current epoch and loss trends
        if epoch > 1:
            block_trend = "↓" if avg_block < loss_history['block'][-2] else "↑"
            rxn_trend = "↓" if avg_rxn < loss_history['rxn'][-2] else "↑"
            total_trend = "↓" if avg_total < loss_history['total'][-2] else "↑"
            print(f"Epoch {epoch:03d} | block_loss={avg_block:.4f} {block_trend} rxn_loss={avg_rxn:.4f} {rxn_trend} total={avg_total:.4f} {total_trend}")
        else:
            print(f"Epoch {epoch:03d} | block_loss={avg_block:.4f} rxn_loss={avg_rxn:.4f} total={avg_total:.4f}")
        
        # NEW: Print complete loss history for all epochs
        print(f"  Complete loss history (all epochs):")
        print(f"    block: {loss_history['block']}")
        print(f"    rxn: {loss_history['rxn']}")
        print(f"    total: {loss_history['total']}")

        if scheduler is not None:
            scheduler.step()

        if avg_total < best_loss:
            best_loss = avg_total
            save_obj: Dict[str, object] = {
                "model_state": model.state_dict(),
                "block_vocab": dataset.block_vocab.to_json(),
                "rxn_vocab": dataset.rxn_vocab.to_json(),
                "hidden_dim": args.hidden_dim,
                "num_gnn_layers": args.num_gnn_layers,
            }
            torch.save(save_obj, args.save)
            print(f"  saved: {args.save}")


if __name__ == "__main__":
    train()
