# Enhancing LLM Training via Spectral Clipping [ICML 2026]

[![arXiv](https://img.shields.io/badge/arXiv-2603.14315-b31b1b.svg)](https://arxiv.org/pdf/2603.14315)
[![citation](https://img.shields.io/badge/cite-BibTeX-blue.svg)](#citation)

This repository contains the implementation of the framework of **[SPECTRA](https://arxiv.org/pdf/2603.14315)** for LLM training, built on top of [llm-baselines](https://github.com/epfml/llm-baselines) and [nanoGPT](https://github.com/karpathy/nanogpt).

**Authors:** Xiaowen Jiang, Andrei Semenov, Sebastian U. Stich

## Overview

**SPECTRA** (**SPE**ctral **C**lipping for LLM **TR**aining **A**cceleration) is a general framework that applies *soft spectral clipping* to optimizer updates (and optionally to raw stochastic gradients) during LLM training to improve convergence and generalization. It works with any  optimizer that uses decoupled weight decay (AdamW, Signum, AdEMAMix, etc.)

**Implementation of SPECTRA as a wrapper**
([`src/optim/spectra.py`](src/optim/spectra.py)):
 After the base optimizer computes $X_{k+1}$, SPECTRA recovers $U_k$ from the weight difference, applies soft spectral clipping, and recomputes the weights:
 $X_{k+1} = (1 - \lambda \eta_k) X_k - \alpha \eta_k \cdot H_c(U_k)$ with $U_k = (X_k - X_{k+1})/\eta_k - \lambda X_k$,
where 
$H_c(X) = \left(I + XX^T/c^2\right)^{-1/2} X$
maps each singular value $\sigma \mapsto \sigma / \sqrt{1 + \sigma^2 / c^2}$, which smoothly clips large singular values toward $c$ while preserving small ones. The matrix inverse square root is computed via Newton-Schulz iteration([`src/optim/post_process.py`](src/optim/post_process.py)).

This design is optimizer-agnostic and requires no modifications to the internals of the base optimizer. (However, it is recommended to integrate SPECTRA directly into the base optimizer to eliminate redundant computation.)

## Quick Start

### 1. Create Environment

We recommend using [uv](https://docs.astral.sh/uv/) for fast environment setup:

```bash
uv venv spectra --python 3.10
source spectra/bin/activate
uv pip install -r requirements.txt
```

Alternatively, using conda/pip:

```bash
conda create -n spectra python=3.10
conda activate spectra
pip install -r requirements.txt
```

### 2. Run a Simple Example

Train a small LLaMA model with SPECTRA-AdamW on the slimpajama6B dataset using one gpu (with `--spectral_post_process clip` flag):

```bash
python ./src/main.py --config_format base --model llama --opt adamw --spectral_post_process clip
```

## Reproducibility
We [`present`](/scripts) scripts for reproducing our results for 160M, 780M and 820M dense Llama-based models.

For SVD recording and analysis, we refer to 
[`scripts/records/adamw_sigvals_noise.sh`](scripts/records/adamw_sigvals_noise.sh) as an example of how to record the singular value distribution of raw stochastic gradients and their noise during training.

To evaluate trained checkpoints on standard benchmarks using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), we refer to [`src/eval/run_eval.py`](src/eval/run_eval.py) and [`scripts/eval`](scripts/eval).


## More Available Hyperparameters
We introduce a few SPECTRA-related hyperparameters here; the remaining ones can be found in [`src/config/base.py`](src/config/base.py). 

### Spectral Post-Processing (Applied to Updates)

| Parameter | Default | Description |
|-------------|---------|-------------|
| `--spectral_post_process` | `none` | Post-processing mode: `none`, `clip` (soft spectral clipping), or `normalize` (Muon-style, all SVs mapped to ~1) |
| `--spectral_clip_c` | `10.0` | Base clipping threshold $c$ — singular values much larger than $c$ are smoothly clipped toward $c$ |
| `--spectral_ns_steps` | `10` | Number of Newton-Schulz iterations for computing the matrix inverse square root |
| `--spectral_apply_to` | `all` | Which parameters to post-process: `2d` (weight matrices only) or `all` (including vectors and biases) |

### Clipping Threshold Schedule

By default, the clipping threshold $c$ is adjusted dynamically during warmup phase so that $c \cdot \eta = \text{const}$, and stays constant afterwards. This default behaviour was sufficient for all experiments in our paper, and we recommend it as the starting point.

Other clipping-threshold dynamics — disabling the warmup adjustment, decaying $c$ in a final phase, or shaping the decay with a custom curve — are also available in [`src/config/base.py`](src/config/base.py).


### Spectral Gradient Pre-Clipping

In addition to post-processing optimizer updates, SPECTRA supports per-parameter spectral clipping of raw gradients as a pre-processing step, applied before the standard global L2 gradient clipping (`--grad_clip`). 
(Not recommended when post spectral-clipping is turned on.)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--spectral_grad_clip` | `none` | Pre-clipping mode for raw gradients: `none` or `clip` |
| `--spectral_grad_clip_c` | `0.1` | Clipping threshold for spectral gradient clipping |

Similar to post-clipping, various clipping-threshold dynamics for pre-clipping are also available for exploration. 

**Order of operations in the training loop:**
1. Backward pass (compute gradients)
2. Spectral gradient clipping (per-parameter, if enabled)
3. Global L2 gradient clipping (`--grad_clip`)
4. Optimizer step (+ spectral update post-processing, if enabled)


### Shared Memory Data Loading

By default, data is read from disk via `np.memmap` on every batch. This avoids high memory usage but can become a bottleneck when disk I/O is slow (e.g., network filesystems on HPC clusters) or when datasets are large (e.g., FineWeb-Edu at 187GB). 

If disk I/O is slow, the `--shared_memory` flag is recommended to be enabled: rank 0 loads the dataset into POSIX shared memory once (incurring a one-time loading cost), and all GPU workers attach to the same region (zero-copy) — fast reads without memory duplication.


## Contact & Citation

Please do not hesitate to reach out to us if you have questions. Feel free to open an [issue](../../issues). If you find this work useful, please consider citing the paper:

<a id="citation"></a>
```bibtex
@inproceedings{jiang2026spectra,
  title={{Enhancing LLM Training via Spectral Clipping}},
  author={Jiang, Xiaowen and Semenov, Andrei and Stich, Sebastian U},
  booktitle={Forty-third International Conference on Machine Learning},
  url={https://arxiv.org/abs/2603.14315},
  year={2026},
}
```
