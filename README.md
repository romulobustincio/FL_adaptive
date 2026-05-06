# QFL-Adaptive: Federated Adaptive Communication for Pole-Angle QFL Architectures

This repository contains the experimental artifacts for the paper:

**QFL-Adaptive: Uma Abordagem Híbrida de Aprendizado Federado Quântico Personalizado e Resiliente**

Accepted at **SBRC 2026**.

## Overview

QFL-Adaptive is a federated learning framework designed to evaluate adaptive communication, selective aggregation, and personalization in architectures inspired by Quantum Federated Learning (QFL).

The main idea is to separate the model into two functional parameter blocks:

- `phi` (`ϕ`): the larger parametrizable body of the model.
- `theta` (`θ`): the smaller decision/transmission head used in reduced communication mode.

When the communication channel is stable, the client sends the complete update `(ϕ, θ)`. When the channel is degraded, the client sends only the reduced update `(θ)`. This allows the server to preserve partial client contributions instead of discarding the entire round.

## Important Methodological Note

The current implementation is a **differentiable emulation** of a pole-angle segmented QFL architecture.

It does **not** execute physical quantum circuits, quantum gates, quantum measurements, or real quantum hardware. It also does **not** simulate physical quantum noise such as decoherence, depolarizing noise, gate errors, or measurement errors.

The experiments focus on the **federated systems layer** of QFL-Adaptive:

- adaptive uplink communication;
- partial parameter transmission;
- selective aggregation;
- personalized model interpolation;
- robustness under stochastic communication-channel degradation.

Validation on real quantum hardware or Qiskit-based quantum circuit simulation is left as future work.

## Repository Structure

```text
FL_adaptive/
├── main_qfl.py              # Main experiment runner
├── qfl_core.py              # Core model, client, server, and aggregation logic
├── run_experiments.sh       # Script for batch execution
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── results/                 # Aggregated experimental outputs
├── figures/                 # Figures generated from experiments
└── configs/                 # Optional configuration files
```

If some folders are missing after cloning the repository, they can be created manually:

```bash
mkdir -p results figures configs
```

## Requirements

Recommended environment:

- Python 3.10 or newer
- PyTorch
- torchvision
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn

Install dependencies with:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available yet, create it with at least:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
torchvision
```

## Quick Start

Run a single experiment:

```bash
python main_qfl.py \
  --dataset breast \
  --channel_threshold 0.7 \
  --bad_channel_prob 0.8 \
  --rounds 25 \
  --clients 10 \
  --seed 43 \
  --output_dir results
```

Run the full experimental batch:

```bash
bash run_experiments.sh
```

## Experimental Protocol

The experiments evaluate four binary classification scenarios:

- `breast`
- `fashion`
- `mnist`
- `pneumonia`

The main federated setup uses:

| Component | Value |
|---|---|
| Number of clients | 10 |
| Communication rounds | 25 |
| Local epochs | 1 |
| Optimizer | Adam |
| Learning rate | 0.005 |
| Personalization factor | `alpha = 0.8` |
| Channel thresholds | `tau ∈ {0.3, 0.7}` |
| Bad-channel probabilities | `p ∈ {0.0, 0.2, 0.5, 0.8}` |

## Communication-Channel Model

The channel degradation is modeled stochastically.

For each client and communication round, the channel may be degraded with probability `p`.

When the channel is degraded, the client sends only the reduced parameter block `theta`. Otherwise, it sends the full model update.

Conceptually:

```text
Good channel      -> transmit full update: phi + theta
Degraded channel  -> transmit reduced update: theta only
```

This models communication instability, congestion, or degraded uplink conditions. It does not model physical quantum noise.

## Main Metrics

The experiments report the following metrics.

### Global Accuracy

Accuracy over the test set.

### Bandwidth Saving

Reduction in transmitted uplink parameters compared to always transmitting the full model.

### Retention Index

Accuracy preservation under degraded communication relative to the ideal communication setting.

```text
Retention = Accuracy(p) / Accuracy(p = 0)
```

## Reproducing the Paper Results

To reproduce the complete set of experiments reported in the paper, run:

```bash
bash run_experiments.sh
```

The script should execute combinations of:

```text
datasets = {breast, fashion, mnist, pneumonia}
thresholds = {0.3, 0.7}
bad_channel_probabilities = {0.0, 0.2, 0.5, 0.8}
seeds = {41, 42, 43, 44, 45}
```

Expected outputs include:

```text
results/
├── raw_results.csv
├── summary_by_dataset.csv
├── summary_by_threshold.csv
└── summary_resilience_tau07.csv

figures/
├── heatmap_accuracy.png
├── heatmap_savings.png
├── radar_breast.png
├── radar_fashion.png
├── radar_mnist.png
├── radar_pneumonia.png
└── accuracy_bars_tau07.png
```

## Notes on Random Seeds

For reproducibility, the experiments should be run with fixed random seeds.

The paper reports results using five seeds. If using this repository to reproduce the reported numbers, verify that `run_experiments.sh` iterates over the same seeds used in the final experiments.

Example:

```bash
SEEDS=(41 42 43 44 45)
```

Each output row should include the corresponding seed value.

## Artifact Scope

This repository provides:

- source code for the QFL-Adaptive emulated framework;
- scripts for running experiments;
- aggregated experimental results;
- figure-generation outputs;
- configuration details required for replication.

This repository does not provide:

- execution on real quantum hardware;
- Qiskit-based physical quantum circuit simulation;
- quantum gate noise simulation;
- decoherence or measurement-error simulation.

## How to Interpret the Results

The results should be interpreted as an evaluation of the communication and aggregation behavior of QFL-Adaptive under stochastic channel degradation.

The experiments support claims about:

- reduced uplink communication cost;
- robustness to partial client updates;
- selective aggregation of segmented parameter blocks;
- personalized federated interpolation.

The experiments do not claim quantum advantage or physical quantum hardware validation.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{Bustincio2026QFLAdaptive,
  author    = {Bustincio, Romulo W. C. and Condori Pozo, Edgar and Hancco Ancori, Ricardo J. and Silva, Francisco Airton and de Souza, Allan M. and Bittencourt, Luiz Fernando},
  title     = {{QFL-Adaptive}: Uma Abordagem Híbrida de Aprendizado Federado Quântico Personalizado e Resiliente},
  booktitle = {Anais do Simpósio Brasileiro de Redes de Computadores e Sistemas Distribuídos},
  year      = {2026},
  publisher = {SBC}
}
```

## License

Add a license file to clarify reuse conditions.

Recommended options:

- MIT License, for permissive reuse.
- Apache License 2.0, for permissive reuse with patent clauses.
- CC BY 4.0, for documentation and non-code artifacts.

## Contact

For questions about the artifact, please contact:

```text
romulo.bustincio@ic.unicamp.br
```

## Reproducibility Statement

The artifact was prepared to support transparency and replicability of the experimental results reported in the paper. The implementation is intentionally lightweight and reproducible, focusing on the federated communication policy and selective aggregation mechanism of QFL-Adaptive.
