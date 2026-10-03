# PCNet

Welcome to the official repository of **PCNet**: **P**eriodic **C**ycle **Net**work for Efficient Multivariate Time Series Forecasting.

## Introduction

**PCNet** is a unified framework for multivariate time series forecasting that leverages periodic patterns through learnable templates. It integrates multiple forecasting strategies into a single model, allowing flexible switching between different modes via the `pcnet_mode` parameter.

### Model Variants

PCNet supports 5 different modes, combining two key design choices:

| Mode | Template Type | Forecast Strategy | Description |
|:----:|:-------------:|:-----------------:|:-----------:|
| **DPCF** | Dynamic (Attention-based) | Concatenation | Dynamic periodic template with concatenation fusion |
| **DPRF** | Dynamic (Attention-based) | Residual | Dynamic periodic template with residual forecasting |
| **FPCF** | Fixed (Learnable) | Concatenation | Fixed periodic template with concatenation fusion |
| **FPRF** | Fixed (Learnable) | Residual | Fixed periodic template with residual forecasting |
| **MLP** | Fixed (Learnable) | Residual | Simple MLP baseline with fixed template |

### Key Features

- **Periodic Template Learning**: Uses learnable vectors aligned with data periodicity (hourly, daily, weekly cycles)
- **Dynamic Template Generation**: DPCF/DPRF modes use cross-attention to generate input-aware dynamic templates
- **Lightweight Architecture**: Single-layer attention + MLP design with ~1M parameters
- **Unified Framework**: All variants share the same codebase, switchable via `--pcnet_mode`

![PCNet Architecture](Figures/PCNet_Architecture.png)

## Getting Started

### Environment Requirements

```bash
conda create -n PCNet python=3.8
conda activate PCNet
pip install -r requirements.txt
```

### Data Preparation

Download datasets from [Google Drive](https://drive.google.com/file/d/1bNbw1y8VYp-8pkRTqbjoW-TA-G8T0EQf/view) (same as Autoformer, SCINet, etc.).

Create a `./dataset` directory and place all CSV/NPZ files directly:
```
./dataset/ETTh1.csv
./dataset/ETTh2.csv
./dataset/ETTm1.csv
./dataset/ETTm2.csv
./dataset/electricity.csv
./dataset/weather.csv
./dataset/solar_AL.txt
./dataset/traffic.csv
./dataset/PEMS03.npz
./dataset/PEMS04.npz
./dataset/PEMS07.npz
./dataset/PEMS08.npz
```

### Training Scripts

#### Performance Evaluation (DPCF Mode)

Run all dataset experiments for the main DPCF variant:
```bash
sh scripts/run_performance.sh
```

This executes experiments across 12 datasets with different prediction lengths (96/192/336/720 for ETT datasets, 12/24/48/96 for PEMS datasets).

#### Ablation Study

Run all model variants (DPCF, DPRF, FPCF, FPRF, MLP) for ablation analysis:
```bash
sh scripts/run_ablation.sh
```

#### Individual Dataset Scripts

You can also run experiments for specific datasets or variants:
```bash
# Performance evaluation for specific dataset
sh scripts/PCNet/etth1.sh
sh scripts/PCNet/electricity.sh

# Ablation study for specific variant
sh scripts/Ablation/DPRF.sh
sh scripts/Ablation/FPCF.sh
```

### Quick Reproduction

Reproduce all main results with a single command:
```bash
conda create -n PCNet python=3.8
conda activate PCNet
pip install -r requirements.txt
sh scripts/run_performance.sh
```

### Command Line Arguments

Key arguments for PCNet:

| Argument | Description | Default |
|:--------:|:-----------:|:-------:|
| `--model` | Model name | `PCNet` |
| `--pcnet_mode` | Model variant | `DPCF` |
| `--cycle` | Period length | `24` |
| `--template_num` | Number of templates | `8` |
| `--seq_len` | Input sequence length | `96` |
| `--pred_len` | Prediction length | `96` |
| `--enc_in` | Number of input channels | `7` |
| `--d_model` | Model dimension | `512` |
| `--dropout` | Dropout rate | `0.5` |

Example:
```bash
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model PCNet \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --cycle 24 \
  --template_num 8 \
  --pcnet_mode DPCF \
  --train_epochs 30 \
  --patience 5 \
  --dropout 0.5 \
  --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed 2024
```

## Datasets

PCNet is evaluated on 12 real-world datasets:

| Dataset | Variables | Cycle | Prediction Lengths |
|:-------:|:---------:|:-----:|:------------------:|
| ETTh1 | 7 | 24 | 96, 192, 336, 720 |
| ETTh2 | 7 | 24 | 96, 192, 336, 720 |
| ETTm1 | 7 | 96 | 96, 192, 336, 720 |
| ETTm2 | 7 | 96 | 96, 192, 336, 720 |
| Electricity | 321 | 168 | 96, 192, 336, 720 |
| Weather | 21 | 144 | 96, 192, 336, 720 |
| Solar | 137 | 144 | 96, 192, 336, 720 |
| Traffic | 862 | 168 | 96, 192, 336, 720 |
| PEMS03 | 358 | 288 | 12, 24, 48, 96 |
| PEMS04 | 307 | 288 | 12, 24, 48, 96 |
| PEMS07 | 883 | 288 | 12, 24, 48, 96 |
| PEMS08 | 170 | 288 | 12, 24, 48, 96 |

## Project Structure

```
PCNet/
├── models/
│   └── PCNet.py              # Unified PCNet model
├── data_provider/
│   ├── data_factory.py       # Data loading factory
│   └── data_loader.py        # Dataset implementations
├── exp/
│   └── exp_main.py           # Training/testing pipeline
├── scripts/
│   ├── run_performance.sh    # Main performance evaluation
│   ├── run_ablation.sh       # Ablation study runner
│   ├── PCNet/                # Per-dataset scripts (DPCF)
│   │   ├── etth1.sh
│   │   ├── etth2.sh
│   │   └── ...
│   └── Ablation/             # Per-variant scripts
│       ├── DPRF.sh
│       ├── FPCF.sh
│       ├── FPRF.sh
│       └── MLP.sh
├── run.py                    # Main entry point
└── requirements.txt          # Dependencies
```

## Citation

If you find this repo useful, please consider citing our work once it is published.

```bibtex
// TODO: update citation upon acceptance
```

## Acknowledgement

We extend our heartfelt appreciation to the following GitHub repositories for providing valuable code bases and datasets:

- https://github.com/ACAT-SCUT/CycleNet
- https://github.com/ACAT-SCUT/TQNet
- https://github.com/lss-1138/SparseTSF
- https://github.com/thuml/iTransformer
- https://github.com/yuqinie98/patchtst
- https://github.com/cure-lab/LTSF-Linear
- https://github.com/ts-kim/RevIN