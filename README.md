# Anonymous code

This repository contains the main utility experiment code for DP-BiSGD and DP-BiSGD-HF, together with the Offline LiRA membership-inference audit code. The utility and attack experiments follow different training protocols, but share the unified dependency file in the repository root. Do not mix parameters between the two experiment protocols. All experiment parameters can be configured according to the paper.

## 1. Repository Structure

```text
DP-BiSGD/
|-- utility/                  # DP-BiSGD and DP-BiSGD-HF utility experiments
|   |-- algorithm/            # Algorithms
|   |-- data/                 # Dataset loading
|   |-- model/                # Training models
|   |-- privacy_analysis/     # Privacy accounting
|   |-- tools/                # Experiment entry points
|   `-- utils/                # DP optimizers and branch sampling
|-- attack/                   # Membership-inference audits
|-- scripts/
|   |-- prepare_attack.py     # Prepare local attack configurations
|   `-- download_data.py      # Download datasets
|-- requirements.txt          # Unified dependencies
|-- README.md                 # Chinese documentation
`-- README_EN.md              # English documentation
```

## 2. Environment Setup

The utility and attack experiments use the root-level `requirements.txt`. Python 3.10 is recommended. Create an isolated environment for reproduction:

```powershell
conda create -n dpbisgd python=3.10 -y
conda activate dpbisgd
python -m pip install --upgrade pip
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 3. Single Formal Utility Run

Use the parameter settings reported in the paper. The following example runs DP-BiSGD on CIFAR-10 with `epsilon=1`.

After setting up the environment, enter the utility directory and run:

```powershell
python tools/run_dpbisgd_realized_coin_filter.py `
  --algorithm DP-BiSGD `
  --dataset_name CIFAR-10 `
  --epsilon 1 `
  --delta 1e-5 `
  --sigma_small 10.45 `
  --sigma_large 22 `
  --p_large 0.05 `
  --batch_size 8192 `
  --C_t 0.1 `
  --lr 4.0 `
  --momentum 0.9 `
  --large_step_update sgd_bypass_scaled `
  --large_step_lr_scale 0.1 `
  --max_updates_safety 100000 `
  --seed 20260816 `
  --branch_seed 21260819 `
  --device cuda `
  --pld_discretization 1e-4 `
  --pld_log_mass_truncation -50 `
  --pld_tail_mass_truncation 1e-15 `
  --save_model `
  --output_dir outputs/formal/cifar10_dpbisgd_eps1
```

Each run must use a new `--output_dir`. Existing output directories are never overwritten.

DP-BiSGD-HF additionally requires `--use_scattering --input_norm GroupNorm --num_groups 27`. The following example runs DP-BiSGD-HF on FMNIST with `epsilon=1`:

```powershell
python tools/run_dpbisgd_realized_coin_filter.py `
  --algorithm DP-BiSGD-HF `
  --dataset_name FMNIST `
  --epsilon 1 `
  --delta 1e-5 `
  --sigma_small 3.20 `
  --sigma_large 16 `
  --p_large 0.20 `
  --batch_size 2048 `
  --C_t 0.1 `
  --lr 4.0 `
  --momentum 0.9 `
  --large_step_update sgd_bypass_scaled `
  --large_step_lr_scale 0.1 `
  --max_updates_safety 100000 `
  --seed 20260816 `
  --branch_seed 21260819 `
  --device cuda `
  --use_scattering `
  --input_norm GroupNorm `
  --num_groups 27 `
  --pld_discretization 1e-4 `
  --pld_log_mass_truncation -50 `
  --pld_tail_mass_truncation 1e-15 `
  --save_model `
  --output_dir outputs/formal/fmnist_dpbisgd_hf_eps1
```

## 4. Three-Stage Parameter Search

All three-stage searches use `epsilon=1` and `delta=1e-5`. With `--mode all`, the runner sequentially searches `rho_s=sigma_small/sigma_base`, `rho_l=sigma_large/sigma_base`, and `p_large`.

### 4.1 DP-BiSGD

CIFAR-10:

```powershell
python tools/run_dpbisgd_parameter_impact.py `
  --profile cifar10-dpbisgd --mode all --sigma_base 11 `
  --epsilon 1 --delta 1e-5 `
  --small_ratios 0.50 0.60 0.70 0.80 0.90 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 `
  --large_multipliers 2 3 4 5 `
  --p_values 0.01 0.05 0.10 0.15 0.20 0.25 0.30 `
  --initial_sigma_large_multiplier 4 --initial_p_large 0.05 `
  --workers 1 --seed 20260816 --branch_seed 21260819 `
  --device cuda --output_dir outputs/search/cifar10_dpbisgd
```

### 4.2 DP-BiSGD-HF

CIFAR-10:

```powershell
python tools/run_dpbisgd_hf_parameter_impact.py `
  --profile cifar10-dpbisgd-hf --mode all --sigma_base 11 `
  --epsilon 1 --delta 1e-5 `
  --small_ratios 0.50 0.60 0.70 0.80 0.90 `
  --large_multipliers 2 3 4 5 `
  --p_values 0.01 0.05 0.10 0.15 0.20 0.25 0.30 `
  --initial_sigma_large_multiplier 4 --initial_p_large 0.05 `
  --workers 1 --seed 20260816 --branch_seed 21260819 `
  --device cuda --output_dir outputs/search/cifar10_dpbisgd_hf
```

## 5. Attack Experiments

The membership-inference audit uses a different protocol from the utility experiments. The current protocol covers CIFAR-10 with `epsilon=1,2,4`. For each method and privacy budget, it trains one target model and eight reference models.

For example, run the two `epsilon=1` attack audits as follows:

```powershell
python -m attacks.lira.run_lira_pipeline --config configs/local/reproduction01/eps1_gaussian.json
python -m attacks.lira.run_lira_pipeline_coin_aware_pld --config configs/local/reproduction01/eps1_dpbisgd.json
```

The main attack table reports the `official_logpdf_fixed` score.

