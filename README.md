# SPOT

SPOT is a synthesis-first offline-to-online framework for target-specific and synthesizable molecular generation. Instead of starting only from protein sequence or structure and then generating molecules that may be chemically unrealistic or hard to synthesize, SPOT begins with a synthesis prior learned from feasible reaction trajectories and then adapts that prior toward a specific target.

The central idea is to reconcile two objectives that are often separated in molecular design workflows:

- learning how drug-like molecules can be constructed through feasible synthesis routes
- reallocating generation probability toward molecules that are compatible with a target protein pocket

In this repository, SPOT combines:

- forward synthesis trajectories extracted from retrosynthesis routes
- synthesis-aware policy pretraining over building blocks and reaction templates
- offline-to-online trajectory-level fine-tuning under protein-conditioned reward
- terminal pocket evaluation with PLANTAIN and optional Vina refinement

The overall goal is to generate molecules that remain synthesizable while becoming increasingly target-specific during fine-tuning.

## Overview

Target-specific molecular generation is commonly driven by protein sequence or structure, but the resulting molecules are often difficult to synthesize, limiting their practical value in drug discovery. SPOT addresses this problem from the opposite direction: it first learns a synthesis prior from forward synthesis trajectories converted from retrosynthetic routes, and then performs protein-conditioned trajectory-level fine-tuning so that the marginal probability of generating a molecule is aligned with its target-related terminal reward.

Starting from purchasable building blocks, the policy selects a building block and a reaction template at each step to produce the next intermediate or terminal molecule. During target adaptation, terminal rewards are derived from pocket-based evaluation using PLANTAIN, with optional Vina refinement, together with lightweight medicinal-chemistry regularization.

The repository supports:

- building a forward synthesis dataset from retrosynthesis routes
- behavior-cloning pretraining of `SynthPolicyNet`
- offline and online trajectory-balance fine-tuning
- protein-conditioned inference with route export
- PLANTAIN and optional Vina reranking of generated leads
- route visualization and metric plotting

## Repository layout

- `SynthPolicyNet/`: behavior-cloning model, datasets, and training code
- `LeadGFlowNet/`: conditional policy, protein encoder, reward logic, and TB training
- `scripts/`: dataset preparation, training wrappers, rescoring, plotting, and visualization
- `PRD/`: project notes and paper-oriented documentation
- `lib/`: local assets and third-party model/runtime dependencies used by the project
- `test/`: example pocket structures and small test assets

## Method summary

### 1. Forward synthesis policy pretraining

Retrosynthesis routes are converted into forward one-step transitions of the form:

- state: current intermediate molecule
- action: `(building block, reaction template)`
- result: next molecule

`SynthPolicyNet` uses graph encoders for molecular states and building blocks, then predicts actions with a factorized policy. The default training path is block-first:

```math
P_F(a_t \mid s_t) = P_\text{block}(b_t \mid B_t)\, P_\text{rxn}(r_t \mid B_t, b_t)
```

The codebase also supports a reaction-first variant during training and online TB fine-tuning.

### 2. Protein-conditioned trajectory-balance fine-tuning

`ConditionalSynthPolicy` conditions the synthesis policy on a protein embedding, using a FiLM-style modulation of the molecular state representation.

The online objective is trajectory balance:

```math
\mathcal{L}_{TB} = \left(\log Z + \sum_t \log P_F - \log R - \sum_t \log P_B\right)^2
```

The implementation supports:

- simple backward approximations
- optional learned backward policies
- open-space exploration through reaction templates and external building blocks

### 3. Reward modeling

Depending on configuration, terminal reward can use:

- PLANTAIN-derived pose score transformed into reward
- Vina-refined docking energy with medicinal chemistry regularization
- optional QSAR reward

QED, SA, and Lipinski-style penalties are available as stabilizing terms. The training code also supports optional per-step shaping in addition to terminal reward.

## Installation

The repository ships with a Conda environment file:

```bash
conda env create -f environment.yml
conda activate phar
```

Core Python dependencies include:

- PyTorch
- PyTorch Geometric
- RDKit
- pandas / numpy / tqdm

Some advanced workflows require extra local tooling that is not fully managed by `environment.yml` alone:

- PLANTAIN assets under `lib/plantain/`
- a local ESM2 model under `lib/models--facebook--esm2_t30_150M_UR50D`
- AutoDock Vina / `vina` Python package for docking refinement
- `obabel` for some ligand preparation paths

## Data preparation

SPOT expects forward synthesis trajectories derived from retrosynthesis routes.

Typical workflow:

```bash
python -m scripts.filter_pactivity_curate \
  --input data/protein_ligand_pactivity.csv \
  --output data/protein_ligand_pactivity_curated.csv \
  --protein-filter gpcr_kinase \
  --min-qed 0.6 \
  --max-sa 5.0 \
  --cap 30000 \
  --scaffold-dedupe
```

```bash
python -m scripts.build_all_routes_dataset \
  --config config.yml \
  --input data/protein_ligand_pactivity_curated.csv \
  --stock zinc \
  --expansion-policy uspto \
  --filter-policy uspto \
  --output data/reaction_paths_all_routes.csv \
  --max-stock-mw 200
```

```bash
python -m scripts.forward_trajectories \
  --input data/reaction_paths_all_routes.csv \
  --output data/forward_trajectories.csv \
  --skip-start-steps \
  --max-block-mw 200
```

Main generated artifacts:

- `data/reaction_paths_all_routes.csv`
- `data/forward_trajectories.csv`

## Training

### Behavior cloning pretraining

```bash
python -m SynthPolicyNet.train_policy \
  --input data/reaction_paths_all_routes.csv \
  --forward data/forward_trajectories.csv \
  --save checkpoints/synth_policy_net.pt
```

### Offline trajectory balance

```bash
bash scripts/offline_tb_train.sh 50 512
```

Or run the trainer directly:

```bash
python -u LeadGFlowNet/offline_tb_train.py \
  --input data/reaction_paths_all_routes.csv \
  --forward data/forward_trajectories.csv \
  --pactivity data/protein_ligand_pactivity_curated.csv \
  --use-checkpoint \
  --checkpoint checkpoints/synth_policy_net.pt \
  --save checkpoints/leadgflownet_offline_tb.pt
```

### Online trajectory balance

Wrapper script:

```bash
bash scripts/online_tb_train.sh 5
```

Example with docking-guided reward:

```bash
python -u -m LeadGFlowNet.online_tb_train \
  --input data/reaction_paths_all_routes.csv \
  --forward data/forward_trajectories.csv \
  --checkpoint checkpoints/leadgflownet_offline_tb.pt \
  --epochs 1 \
  --episodes-per-epoch 600 \
  --max-steps 8 \
  --use-docking-guidance \
  --docking-model plantain \
  --plantain-pocket test/2y9x/2y9x_pocket.pdb \
  --use-vina \
  --save checkpoints/leadgflownet_online_tb.pt
```

## Inference

QSAR-guided or docking-aware inference is supported through `leadgflownet_infer.py`.

Minimal example:

```bash
python leadgflownet_infer.py \
  --input data/reaction_paths_all_routes.csv \
  --forward data/forward_trajectories.csv \
  --checkpoint checkpoints/leadgflownet_online_tb.pt \
  --protein "YOUR_PROTEIN_SEQUENCE" \
  --num-samples 100 \
  --max-depth 8 \
  --export-ranked \
  --output-ranked-csv runs/leads_ranked.csv
```

Example with PLANTAIN and Vina reranking:

```bash
python leadgflownet_infer.py \
  --input data/reaction_paths_all_routes.csv \
  --forward data/forward_trajectories.csv \
  --checkpoint checkpoints/leadgflownet_online_tb.pt \
  --protein "YOUR_PROTEIN_SEQUENCE" \
  --num-samples 50 \
  --max-depth 6 \
  --export-ranked \
  --output-ranked-csv runs/leads_plantain_vina.csv \
  --use-plantain \
  --plantain-pocket test/2y9x/2y9x_pocket.pdb \
  --use-vina \
  --vina-pdbqt-dir runs/vina_pdbqt
```

Typical outputs:

- `runs/lead_routes.json`
- ranked lead CSVs
- optional PLANTAIN pose SDFs
- route visualization assets

## Visualization and analysis

Route visualization:

```bash
python scripts/visualize_routes_json.py \
  --json runs/lead_routes.json \
  --out runs/routes_viz \
  --max-routes 50 \
  --write-dot
```

Lead statistics:

```bash
python scripts/plot_infer_leads.py \
  --json runs/lead_routes.json \
  --out runs/plots_infer
```

## Notes

- The repository currently mixes research code, generated artifacts, and local runtime assets. For a clean public release, you may want to trim large runtime outputs under `runs/`.
- Some scripts assume local file layouts in `lib/` and `test/`; check paths before running them on a new machine.
- `PRD/paper.md` contains a paper-oriented description of the method and has been aligned with the current code paths.

## Citation

If you use this repository in a paper or report, cite the project together with the associated method description in `PRD/paper.md`.
