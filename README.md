# BitNet-GPT2-125M

An educational GPT-2 (125M) quantization-aware training experiment that replaces GPT-2's MLP projection layers with custom ternary `BitLinear` layers.

The trainable parameters remain floating-point tensors, but every `BitLinear` forward pass quantizes its weight tensor to `{-1, 0, 1}`. A straight-through estimator (STE) lets gradients update the underlying floating-point weights during training.

> **Important:** this repository demonstrates ternary QAT. It does **not** currently pack weights into a physical 1.58-bit representation or provide optimized ternary CUDA/CPU kernels. The checkpoint remains a normal floating-point PyTorch/Safetensors checkpoint.

## Repository layout

| File | Purpose |
| --- | --- |
| `model_converter.py` | Defines `BitNetSTE` and `BitLinear`, converts GPT-2 MLP projections, saves converted models, and loads trained checkpoints. |
| `trainer.py` | Complete WikiText-2 QAT training script with periodic generation and checkpoint saving. |
| `run.py` | Interactive inference using either the hosted trained checkpoint or a local checkpoint directory. |
| `requirements.txt` | Project dependencies. |

The original trained checkpoint is hosted at:

`123aloo123/BitNet-GPT2-125M-Ternary`

---

## Installation

```bash
git clone https://github.com/MEESAM749/BitNet-GPT2-125M.git
cd BitNet-GPT2-125M

python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run the trained model

The simplest way to use the project is:

```bash
python run.py
```

By default, `run.py` loads:

```text
123aloo123/BitNet-GPT2-125M-Ternary
```

The Hugging Face config, tokenizer, and checkpoint are downloaded automatically when needed.

Example:

```text
Model loaded on cuda. Type 'quit' or 'exit' to stop.

User Prompt: The future of artificial intelligence is

BitNet: ...
```

You can also load a local checkpoint:

```bash
python run.py --model ./bitnet_final_2hr
```

Useful options:

```bash
python run.py \
    --model ./bitnet_final_2hr \
    --max-new-tokens 80 \
    --temperature 0.8
```

---

# Train the model

`trainer.py` is now a complete training entry point. It:

1. loads pretrained GPT-2;
2. replaces GPT-2 MLP `c_fc` and `c_proj` projections with `BitLinear`;
3. preserves the pretrained projection weights and biases;
4. loads WikiText-2;
5. performs ternary quantization-aware training;
6. periodically generates a qualitative sample;
7. saves periodic checkpoints;
8. saves a final model that can be loaded by `run.py`.

To run the original two-hour style experiment:

```bash
python trainer.py
```

Default configuration:

```text
Base model:             gpt2
Dataset:                wikitext / wikitext-2-raw-v1
Duration:               2 hours
Batch size:             4
Sequence length:        128
Learning rate:          5e-5
Showcase interval:      15 minutes
Checkpoint interval:    30 minutes
Final output:           ./bitnet_final_2hr
```

Example shorter run:

```bash
python trainer.py \
    --duration-hours 0.25 \
    --batch-size 4 \
    --output-dir ./bitnet_test_run
```

For a lower-memory run, use gradient accumulation:

```bash
python trainer.py \
    --batch-size 1 \
    --gradient-accumulation-steps 4
```

See all options with:

```bash
python trainer.py --help
```

### Padding is excluded from the training loss

GPT-2 does not define a pad token by default, so the script uses EOS as the tokenizer pad token. Padded positions are changed to `-100` in the labels so they are ignored by the language-model loss instead of being learned as repeated EOS tokens.

---

# Convert ordinary GPT-2 without training

To create a QAT-ready ternary GPT-2 checkpoint without starting training:

```bash
python model_converter.py \
    --base-model gpt2 \
    --output-dir ./bitnet_gpt2_init
```

This converts the GPT-2 MLP projections to `BitLinear`, copies the pretrained weights and biases, and saves the result.

It does **not** perform QAT by itself.

---

# Use the architecture from Python

## Convert pretrained GPT-2

```python
from model_converter import convert_pretrained_model

model = convert_pretrained_model("gpt2")
```

Or convert an existing compatible model in place:

```python
from transformers import AutoModelForCausalLM
from model_converter import perform_surgery

model = AutoModelForCausalLM.from_pretrained("gpt2")
perform_surgery(model)
```

## Load a trained BitNet checkpoint

```python
from model_converter import load_bitnet_model

model, tokenizer, device = load_bitnet_model(
    "123aloo123/BitNet-GPT2-125M-Ternary"
)
```

A local path works too:

```python
model, tokenizer, device = load_bitnet_model("./bitnet_final_2hr")
```

---

# How the ternary layer works

The forward pass performs AbsMean ternarization:

```python
gamma = W.abs().mean()
W_quant = torch.clamp(
    torch.round(W / (gamma + 1e-5)),
    min=-1,
    max=1,
)
```

so the effective forward-pass weight values are:

```text
-1, 0, 1
```

The backward pass uses a straight-through estimator:

```python
return grad_output
```

This bypasses the non-differentiable rounding operation during backpropagation and lets the optimizer update the underlying floating-point shadow weights.

The current implementation converts GPT-2's MLP projections:

```text
mlp.c_fc
mlp.c_proj
```

The attention projections remain unchanged.

---

# GPT-2 `Conv1D` and tensor orientation

Hugging Face GPT-2 implements these projections using its `Conv1D` helper, whose stored weight orientation differs from `torch.nn.Linear`.

```text
GPT-2 Conv1D:  [in_features, out_features]
BitLinear:     [out_features, in_features]
```

`model_converter.py` handles this conversion explicitly when performing model surgery.

The checkpoint loader also checks tensor shapes and transposes target projection weights when the stored layout requires it.

---

# Compatibility with the original trained checkpoint

The original experiment created the replacement MLP projections with `bias=False`. The already-published Hugging Face checkpoint therefore does not contain those MLP bias tensors.

The repaired loader inspects the actual checkpoint tensors:

- legacy checkpoint without MLP biases -> reconstructs the original no-bias architecture;
- new checkpoint produced by the repaired trainer -> reconstructs the bias-preserving architecture.

This keeps the original trained model loadable while preventing unnecessary information loss in future training runs.

If you intentionally want to reproduce the original no-bias setup:

```bash
python trainer.py --legacy-drop-mlp-bias
```

or:

```bash
python model_converter.py --drop-mlp-bias
```

---

# What this project is — and is not

This project is useful for learning about:

- ternary weight quantization;
- quantization-aware training;
- straight-through estimators;
- modifying pretrained Transformer architectures in PyTorch;
- checkpoint compatibility when custom layers use a different tensor layout.

However, the trainable weights are still stored as normal floating-point tensors. During inference, `BitLinear` dynamically creates a ternary weight tensor and calls PyTorch's normal linear operation.

So the current implementation does **not** yet provide the memory footprint or inference-speed advantages of a model using packed ternary weights and specialized low-bit kernels.

The 125M-parameter scale is intentionally small and should be treated as an educational/research experiment rather than a competitive modern language model.

---

# Original experiment

The initial experiment trained the converted GPT-2 125M model on WikiText-2 for approximately two hours on an NVIDIA T4 and periodically generated text to observe recovery after the initial ternary conversion.

The current code preserves that experiment's basic idea while separating conversion, training, and inference into working reusable components.

---

## Future work

Potential extensions include:

- packed ternary checkpoint storage;
- optimized CUDA/C++/CPU ternary kernels;
- activation quantization;
- more complete BitNet architectural changes;
- evaluation on standard language-model benchmarks;
- larger models and teacher/student distillation experiments.
