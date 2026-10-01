# mini-torch

A tiny deep learning framework built from scratch in Python on top of NumPy. It covers tensors, reverse-mode autograd, layers, optimizers and a training loop, and works up to CNNs and a small transformer.

The project's scope and module order follow [TinyTorch](https://mlsysbook.ai/tinytorch), but every component is written from scratch here with NumPy as the only numerical backend.

## What will be included

| # | Stage | What it adds |
|---|-------|--------------|
| 01 | Tensor | `Tensor` wrapping `np.ndarray`: arithmetic, broadcasting, matmul, reductions, reshape/transpose |
| 02 | Activations | ReLU, Sigmoid, Tanh, GELU, Softmax |
| 03 | Layers | `Module` base class, `Linear`, `Dropout`, `Sequential`, parameter collection |
| 04 | Losses | MSE, cross-entropy (numerically stable log-softmax) |
| 05 | DataLoader | `Dataset`, `DataLoader` with batching and shuffling |
| 06 | Autograd | Computation graph, `backward()`, gradient functions for every op |
| 07 | Optimizers | SGD (momentum), Adam, AdamW |
| 08 | Training | Training/eval loop, LR scheduling, gradient clipping, checkpointing |
| 09 | Convolutions | `Conv2d`, `MaxPool2d` (im2col) |
| 10 | Tokenization | Character and BPE tokenizers |
| 11 | Embeddings | Token and positional embeddings |
| 12 | Attention | Scaled dot-product and multi-head attention, causal masking |
| 13 | Transformers | LayerNorm, transformer block, GPT-style decoder |
| 14+ | Systems | Profiling, quantization, compression, KV-cache, benchmarking |

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/Scripts/activate     # Windows (Git Bash); use .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
pytest
```
