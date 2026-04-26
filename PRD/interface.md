## PHAR CLI Reference

This document describes the two main command-line interfaces exposed by PHAR:

- online inference with route sampling and lead reranking
- online trajectory-balance fine-tuning

It is intended for service integration, workflow automation, and reproducible research runs. Example paths below use `/home/jb/phar`; replace them with your local deployment paths.

---

## 1. Environment Prerequisites

- Python 3.10+ is recommended.
- Create the environment with the project-level `environment.yml`.
- Prepare output and cache directories before running docking-aware workflows:

```bash
mkdir -p /home/jb/phar/runs \
         /home/jb/phar/runs/vina_pdbqt \
         /home/jb/phar/runs/plantain_poses
```

- If `obabel` is not installed at `/usr/local/bin/obabel`, resolve it dynamically:

```bash
command -v obabel
```

- Docking-aware workflows may additionally require:
  - PLANTAIN assets under `lib/plantain/`
  - a local ESM2 model under `lib/models--facebook--esm2_t30_150M_UR50D`
  - the `vina` Python package
  - OpenBabel and, optionally, ADFR/Meeko for stricter ligand preparation

---

## 2. Inference CLI: `leadgflownet_infer.py`

Program path:

```bash
/home/jb/phar/leadgflownet_infer.py
```

### 2.1 Purpose

The inference CLI samples forward synthesis trajectories from the learned policy and collects terminal molecules. It supports:

- protein-conditioned generation
- QSAR-guided ranking
- PLANTAIN scoring
- optional Vina reranking
- template-guided open-space expansion
- JSON and ranked CSV/JSON export

### 2.2 Core Inputs

- `--protein`: target protein sequence, required
- `--checkpoint`: policy checkpoint, default `checkpoints/synth_policy_net.pt`
- `--input`: retrosynthesis route CSV, default `data/reaction_paths_all_routes.csv`
- `--forward`: forward trajectory CSV, default `data/forward_trajectories.csv`

### 2.3 Main Sampling Controls

- `--num-samples`: number of sampled route trees or paths, default `1000`
- `--max-depth`: maximum route depth, default `10`
- `--branch-block-topk`: top-K block branches per state, default `2`
- `--branch-rxn-topk`: top-K reaction branches per selected block, default `1`
- `--temperature`: sampling temperature, default `1.0`
- `--deterministic`: replace stochastic sampling with deterministic top-k selection
- `--sampling-method {default,nucleus}`: optional nucleus sampling
- `--nucleus-p`: nucleus threshold when `--sampling-method nucleus` is used
- `--expand-mode {tree,path}`: `tree` builds a broader branching structure, while `path` is much faster and samples one branch per depth

### 2.4 Diversity and Export

- `--select-k`: keep at most K final leads after diversity filtering, default `200`
- `--diversity-mode {none,minsim,mmr}`: diversity strategy
- `--mmr-lambda`: score-vs-novelty balance for MMR
- `--export-ranked`: export ranked outputs
- `--output-json`: route JSON, default `runs/lead_routes.json`
- `--output-ranked-csv`: ranked CSV path
- `--output-ranked-json`: ranked JSON path

### 2.5 QSAR Ranking

- `--use-qsar`: enable QSAR-guided ordering
- `--qsar-checkpoint`: QSAR checkpoint path
- `--qsar-mix`: combine policy and QSAR scores as
  `policy_prob^(1-qsar_mix) * sigmoid(qsar)^(qsar_mix)`
- `--min-qsar`: optional post-filter on QSAR sigmoid score

### 2.6 PLANTAIN and Vina Reranking

Enable PLANTAIN:

- `--use-plantain`
- `--plantain-pocket <pocket.pdb>`
- `--plantain-device {auto,cuda,cpu,mps}`
- `--plantain-poses-dir`: output directory for debug SDF poses

Enable Vina reranking:

- `--use-vina`
- `--vina-pdbqt-dir`: ligand PDBQT cache directory
- `--vina-obabel-bin`: OpenBabel executable path
- `--vina-center cx,cy,cz`: optional manual grid center
- `--vina-box-size`: cubic box size in angstroms, default `22`
- `--vina-exhaustiveness`: docking exhaustiveness, default `32`
- `--vina-top-k`: number of PLANTAIN poses refined per ligand, default `3`
- `--vina-full-dock-th`: if optimized affinity is still worse than this threshold, trigger a quick `dock()` pass
- `--vina-strict`: require ADFR/Meeko instead of falling back to OpenBabel
- `--filter-vina-th`: write only rows with `vina_affinity < threshold` to the ranked CSV

### 2.7 Open-Space Expansion

- `--template-walk`: enable template-guided expansion outside the dataset graph
- `--template-csv`: template library path
- `--template-max-rows`: maximum loaded templates
- `--template-try-templates`: maximum templates tried per state
- `--template-sample-blocks`: sampled external blocks for two-reactant templates
- `--extra-blocks-csv`: external building-block pool
- `--extra-blocks-cap`: cap on loaded external blocks
- `--open-max-proposals`: maximum open-space children per state
- `--feasibility-filter {none,rdkit}`: validity filter for open-space products

Inference also supports optional step-wise PLANTAIN reranking of candidate children through arguments such as:

- `--step-plantain`
- `--step-plantain-interval`
- `--step-plantain-topk`
- `--step-plantain-mix`

This is distinct from terminal reranking and acts as a search-time shaping mechanism.

### 2.8 Example: Minimal Ranked Inference

```bash
PYTHONPATH=/home/jb/phar \
python3 /home/jb/phar/leadgflownet_infer.py \
  --input /home/jb/phar/data/reaction_paths_all_routes.csv \
  --forward /home/jb/phar/data/forward_trajectories.csv \
  --checkpoint /home/jb/phar/checkpoints/leadgflownet_online_tb.pt \
  --protein "<PASTE_YOUR_PROTEIN_SEQUENCE_HERE>" \
  --num-samples 50 \
  --max-depth 8 \
  --select-k 20 \
  --export-ranked \
  --output-ranked-csv /home/jb/phar/runs/leads_ranked.csv
```

### 2.9 Example: PLANTAIN + Vina Inference

```bash
PYTHONPATH=/home/jb/phar \
python3 /home/jb/phar/leadgflownet_infer.py \
  --input /home/jb/phar/data/reaction_paths_all_routes.csv \
  --forward /home/jb/phar/data/forward_trajectories.csv \
  --checkpoint /home/jb/phar/checkpoints/leadgflownet_online_tb.pt \
  --protein "<PASTE_YOUR_PROTEIN_SEQUENCE_HERE>" \
  --num-samples 20 \
  --max-depth 6 \
  --select-k 10 \
  --export-ranked \
  --output-ranked-csv /home/jb/phar/runs/leads_plantain_vina.csv \
  --use-plantain \
  --plantain-pocket /home/jb/phar/test/2y9x/2y9x_pocket.pdb \
  --plantain-poses-dir /home/jb/phar/runs/plantain_poses \
  --use-vina \
  --vina-pdbqt-dir /home/jb/phar/runs/vina_pdbqt \
  --vina-obabel-bin "$(command -v obabel)" \
  --vina-box-size 22 \
  --vina-exhaustiveness 16 \
  --vina-top-k 3 \
  --vina-full-dock-th -4.0 \
  --template-walk \
  --template-csv /home/jb/phar/data/top100/template_top100.csv \
  --template-max-rows 300 \
  --template-try-templates 16 \
  --template-sample-blocks 256
```

### 2.10 Output Fields

Common ranked output fields include:

- `smiles`
- `plantain_min`
- `plantain_reward`
- `vina_affinity`
- `vina_affinity_raw`
- `qsar_raw`
- `qsar_sigmoid`
- `qed`
- `new_score`

Ranking behavior depends on the enabled scoring pipeline. When QSAR is enabled, QSAR-guided ordering is applied; when PLANTAIN and Vina are enabled, docking-aware scores are exported for downstream filtering and reranking.

---

## 3. Online TB Training CLI: `LeadGFlowNet/online_tb_train.py`

Program path:

```bash
/home/jb/phar/LeadGFlowNet/online_tb_train.py
```

### 3.1 Purpose

This CLI performs online trajectory-balance fine-tuning of the synthesis policy. It supports:

- QSAR-based terminal reward
- PLANTAIN-based terminal reward
- Vina-refined terminal reward
- medicinal-chemistry shaping terms
- optional per-step shaping
- template-guided and free-connect exploration
- optional learned backward policy

### 3.2 Core Inputs

- `--input`: retrosynthesis route CSV
- `--forward`: forward trajectory CSV
- `--checkpoint`: initialization checkpoint
- `--epochs`
- `--episodes-per-epoch`
- `--max-steps`
- `--lr`
- `--device {auto,cuda,cpu,mps}`
- `--save`: output checkpoint path

### 3.3 Policy and Training Controls

- `--rxn-first`: use reaction-first factorization during sampling
- `--hidden-dim`, `--num-gnn-layers`, `--dropout`
- `--share-encoders`
- `--use-l2-norm`
- `--temperature`

Sampling schedules across epochs:

- `--tf-start`, `--tf-end`: teacher-forcing schedule
- `--temp-start`, `--temp-end`: sampling-temperature schedule
- `--topk-block-start`, `--topk-block-end`
- `--topk-rxn-start`, `--topk-rxn-end`

Other sampling controls:

- `--train-deterministic`
- `--teacher-forcing-prob`
- `--train-branch-block-topk`
- `--train-branch-rxn-topk`
- `--tb-residual-clip`
- `--per-step-retries`

### 3.4 Reward Modes

#### QSAR reward

- `--use-qsar-reward`
- `--qsar-checkpoint checkpoints/qsar.pt`

#### Docking-guided reward

- `--use-docking-guidance`
- `--docking-model plantain`
- `--plantain-pocket /path/to/pocket.pdb`
- `--plantain-device {auto,cuda,cpu,mps}`
- `--plantain-scale`

#### Vina-refined reward

When `--use-vina` is enabled together with docking guidance, the current implementation uses Vina-refined energy as the main reward backbone rather than a small additive adjustment. Useful controls include:

- `--use-vina`
- `--vina-box-size`
- `--vina-exhaustiveness`
- `--vina-top-k`
- `--vina-full-dock-th`
- `--vina-obabel-bin`
- `--vina-strict`
- `--vina-pdbqt-dir`
- `--vina-reward-smooth`: EMA smoothing factor for Vina energy
- `--vina-weight`: reward uses `-vina_weight * E`
- `--prune-bad-vina-th`: optionally skip episodes whose Vina energy is worse than a threshold
- `--prune-mw-th`: optionally skip overly large molecules by molecular weight

### 3.5 Medicinal-Chemistry Shaping

- `--add-qed`: QED bonus weight
- `--sub-sa`: SA penalty weight
- `--lipinski-penalty`: penalty per Lipinski-rule violation
- `--novelty-db <SMI/CSV ...>`
- `--novelty-weight`
- `--use-scaffold-reward`
- `--ref-ligands`
- `--scaffold-weight`

### 3.6 Per-Step Shaping

The training code supports more than terminal reward alone.

- `--use-local-reward`: local structural sanity shaping
- `--local-reward-weight`
- `--perstep-plantain`: enable per-step PLANTAIN shaping
- `--perstep-plantain-interval`
- `--perstep-plantain-weight`

In the current code path, `--use-local-reward` and `--perstep-plantain` are enabled by default unless explicitly changed in the source or wrapper.

### 3.7 Open-Space Exploration

Template-driven exploration:

- `--template-csv`
- `--template-prob`
- `--template-max-rows`
- `--template-try-templates`
- `--template-sample-blocks`
- `--extra-blocks-csv`
- `--extra-blocks-cap`
- `--extra-blocks-prob`
- `--free-walk`
- `--template-walk`

Direct free-connect exploration:

- `--free-connect`
- `--free-connect-prob`
- `--free-connect-tries`
- `--free-connect-sample-blocks`

Mask-relaxed open sampling:

- `--open-eps`
- `--open-topk-block`
- `--open-topk-rxn`
- `--open-temp`
- `--open-max-retries`
- `--feasibility-filter {none,rdkit,onnx}`
- `--feasibility-onnx-path`

### 3.8 Backward Policy Options

The online TB trainer supports multiple backward-policy approximations:

- `--use-backward-policy`: use inbound-degree-style approximation
- `--pb-learned`: learned child-conditioned backward policy
- `--pb-source-aware`: condition backward policy on action source
- `--pb-logsumexp`: log-sum-exp marginalization over candidate parents
- `--pb-candidate-cap`
- `--pb-open-topk-block`
- `--pb-open-topk-rxn`
- `--pb-bc-weight`
- `--pb-buffer-jsonl`
- `--sub-tb-k`: optional sub-trajectory-balance residual

### 3.9 Automatic Reference Collection

PHAR can automatically collect strong Vina-scoring terminal molecules and reuse them as scaffold references in later epochs.

Key options:

- `--auto-ref-vina-th <float>`: collect terminal molecules when `vina_affinity < threshold`
- `--auto-ref-out <path>`: output `.smi` file
- `--auto-ref-use-scaffold`: reuse collected molecules as scaffold references
- `--auto-ref-scaffold-weight <float>`

This is useful for moving from broad search to focused refinement during later training stages.

### 3.10 Example: Docking-Guided Online TB

```bash
PYTHONPATH=/home/jb/phar \
CUDA_VISIBLE_DEVICES=0 \
python3 /home/jb/phar/LeadGFlowNet/online_tb_train.py \
  --input /home/jb/phar/data/reaction_paths_all_routes.csv \
  --forward /home/jb/phar/data/forward_trajectories.csv \
  --checkpoint /home/jb/phar/checkpoints/leadgflownet_offline_tb.pt \
  --epochs 2 \
  --episodes-per-epoch 600 \
  --max-steps 8 \
  --save /home/jb/phar/checkpoints/leadgflownet_online_tb.pt \
  --use-docking-guidance \
  --docking-model plantain \
  --plantain-pocket /home/jb/phar/test/3v8d/3v8d_pocket.pdb \
  --use-vina \
  --vina-pdbqt-dir /home/jb/phar/runs/vina_pdbqt \
  --vina-box-size 22 \
  --vina-exhaustiveness 16 \
  --vina-top-k 1 \
  --vina-full-dock-th -6.0 \
  --vina-reward-smooth 0.9
```

### 3.11 Example: Auto-Reference Collection

```bash
PYTHONPATH=/home/jb/phar \
CUDA_VISIBLE_DEVICES=0 \
python3 /home/jb/phar/LeadGFlowNet/online_tb_train.py \
  --input /home/jb/phar/data/reaction_paths_all_routes.csv \
  --forward /home/jb/phar/data/forward_trajectories.csv \
  --checkpoint /home/jb/phar/checkpoints/leadgflownet_offline_tb.pt \
  --epochs 2 \
  --episodes-per-epoch 600 \
  --max-steps 8 \
  --save /home/jb/phar/checkpoints/leadgflownet_online_tb_autoref.pt \
  --use-docking-guidance \
  --docking-model plantain \
  --plantain-pocket /home/jb/phar/test/3v8d/3v8d_pocket.pdb \
  --use-vina \
  --vina-pdbqt-dir /home/jb/phar/runs/vina_pdbqt \
  --vina-box-size 22 \
  --vina-exhaustiveness 16 \
  --vina-top-k 1 \
  --vina-full-dock-th -6.0 \
  --auto-ref-vina-th -8.0 \
  --auto-ref-out /home/jb/phar/runs/ref_auto_top8.smi \
  --auto-ref-use-scaffold \
  --auto-ref-scaffold-weight 0.2
```

### 3.12 Output Files

Typical outputs include:

- `checkpoints/leadgflownet_online_tb.pt`
- `runs/online_tb_metrics.csv`
- `runs/online_tb_vina_episodes.csv`
- `runs/online_tb_terminals.jsonl`

These files are useful for monitoring convergence, reward distribution, diversity, and docking behavior across epochs.

---

## 4. Practical Notes

- `vina_affinity` is usually the optimized or refined Vina energy and should be preferred over `vina_affinity_raw` for filtering.
- `--vina-full-dock-th` controls when a fast full docking step is triggered after local optimization.
- `--template-walk` in training is effectively an alias for template-guided free-walk exploration.
- In inference, step-wise PLANTAIN reranking and terminal PLANTAIN/Vina reranking are separate mechanisms.
- If you are writing a paper or service wrapper, describe the current implementation as terminal-reward-driven TB fine-tuning with optional per-step shaping, not as a strictly terminal-only system.
