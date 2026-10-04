# BitNet-GPT2-125M

An experimental GPT-2 (125M) project that replaces GPT-2's MLP projection layers with custom ternary `BitLinear` layers and uses a straight-through estimator (STE) so the underlying floating-point weights can still be trained.

> **Important:** this repository demonstrates ternary quantization-aware behavior in PyTorch. It does **not** currently store the model in a packed 1.58-bit format or use custom 1.58-bit inference kernels. The saved checkpoint is still a floating-point Safetensors file.

## What is in this repository?

| File | Purpose | Current status |
| --- | --- | --- |
| `run.py` | Defines `BitNetSTE`, `BitLinear`, performs GPT-2 model surgery, loads a local trained checkpoint, and provides an interactive text-generation loop. | **Usable** once the checkpoint/config/tokenizer files are downloaded. |
| `model_converter.py` | Contains the same ternary layer/surgery logic plus code for downloading/loading the trained checkpoint. | **Experimental / not standalone**: the file references `model` without creating it. |
| `trainer.py` | WikiText-2 training loop with a 2-hour wall-clock limit, checkpointing, and loss logging. | **Experimental / not standalone**: it expects an existing `model` and a `showcase_evolution()` function. |
| `README.md` | Project documentation. | You are here. |

The trained checkpoint is hosted separately on Hugging Face:

`123aloo123/BitNet-GPT2-125M-Ternary`

---

## How the ternary layer works

The repository replaces GPT-2's MLP `c_fc` and `c_proj` modules with a custom `BitLinear` layer.

During each forward pass, the weight tensor is quantized to:

```text
{-1, 0, 1}
```

using AbsMean scaling:

```python
gamma = W.abs().mean()
W_quant = torch.clamp(
    torch.round(W / (gamma + 1e-5)),
    min=-1,
    max=1,
)
```

The backward pass uses a straight-through estimator:

```python
return grad_output
```

so gradients update the underlying floating-point weights even though the forward pass uses ternary weights.

---

# Quick start: run the trained model

This is the most complete path currently provided by the repository.

## 1. Clone the repository

```bash
git clone https://github.com/MEESAM749/BitNet-GPT2-125M.git
cd BitNet-GPT2-125M
```

## 2. Create a Python environment

A virtual environment is recommended.

```bash
python -m venv .venv
```

Activate it using the command appropriate for your operating system, then install the dependencies:

```bash
pip install torch transformers datasets huggingface_hub safetensors
```

## 3. Download the trained checkpoint files

The GitHub repository contains the source code but not the model checkpoint, tokenizer, or GPT-2 config required by `run.py`.

Download the required files from:

`123aloo123/BitNet-GPT2-125M-Ternary`

with:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='123aloo123/BitNet-GPT2-125M-Ternary', local_dir='.', allow_patterns=['config.json','generation_config.json','model.safetensors','tokenizer.json','tokenizer_config.json'])"
```

After downloading, the repository directory should contain at least:

```text
BitNet-GPT2-125M/
├── README.md
├── model_converter.py
├── run.py
├── trainer.py
├── config.json
├── generation_config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

## 4. Start the interactive generator

```bash
python run.py
```

`run.py` will:

1. load the tokenizer and GPT-2 config from the current directory;
2. construct a fresh GPT-2 model;
3. replace each GPT-2 MLP `c_fc` and `c_proj` layer with `BitLinear`;
4. load the trained weights from `model.safetensors`;
5. move the model to CUDA when available, otherwise CPU;
6. start an interactive prompt loop.

Example:

```text
Model loaded successfully! Type 'quit' to exit.

User Prompt: The future of artificial intelligence is

BitNet: ...
```

Type `quit` or `exit` to stop the program.

---

# Use the model from your own Python code

Once the Hugging Face checkpoint files have been downloaded into the repository directory, you can reuse the loader from `run.py`:

```python
import torch
from run import load_local_model

model, tokenizer, device = load_local_model(".")

prompt = "The future of artificial intelligence is"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

# Perform model surgery on normal GPT-2

If you only want to experiment with the ternary architecture and do **not** need the trained checkpoint, use `perform_surgery()` from `run.py`:

```python
from transformers import AutoModelForCausalLM
from run import perform_surgery

model = AutoModelForCausalLM.from_pretrained("gpt2")

perform_surgery(model)

print(model)
```

This replaces GPT-2's MLP projection layers with `BitLinear` layers while copying the original GPT-2 weights into the replacement layers.

This alone does **not** load the trained ternary checkpoint. It only changes the architecture.

---

# Training

`trainer.py` contains the training loop used for the experiment, but in its current repository state it is **not a standalone training command**.

Running:

```bash
python trainer.py
```

directly will not work because the script expects objects that are not created inside `trainer.py`:

- a `model` variable;
- a `showcase_evolution(step, elapsed_minutes)` function.

The training loop itself currently does the following:

- uses `wikitext-2-raw-v1`;
- filters out very short text entries;
- uses `AdamW` with a learning rate of `5e-5`;
- uses a batch size of `4`;
- tokenizes to a maximum length of `128`;
- trains for a maximum wall-clock duration of two hours;
- attempts a text-generation showcase every 15 minutes;
- saves a checkpoint every 30 minutes to `./bitnet_checkpoints/`;
- saves the final model to `./bitnet_final_2hr/`.

If you want to train the model again, the intended flow is:

```text
Load GPT-2
   ↓
Perform BitLinear surgery
   ↓
Place the resulting model on the selected device
   ↓
Run the WikiText QAT loop
   ↓
Save the trained BitLinear checkpoint
```

The current `trainer.py` should therefore be treated as experimental training-loop code that still needs to be connected to model creation and the showcase function before it becomes a clean CLI training script.

---

# About `model_converter.py`

`model_converter.py` defines:

- `BitNetSTE`;
- `BitLinear`;
- `perform_surgery()`.

It also contains code that downloads `model.safetensors` from the Hugging Face checkpoint and attempts to load it.

However, in the current version of the repository, the file attempts to call:

```python
model.load_state_dict(...)
```

without first creating `model`.

Because that code executes at module import time, this also means that:

```python
from model_converter import perform_surgery
```

is **not currently a safe usage pattern**.

For architecture experiments, import `perform_surgery` from `run.py` instead:

```python
from run import perform_surgery
```

---

# Why the README does not use `custom_architecture`

Older documentation for this project showed:

```python
from custom_architecture import perform_surgery
```

There is no `custom_architecture.py` file in this repository.

The actual implementation of `perform_surgery()` is currently present in:

```text
run.py
model_converter.py
```

Because `model_converter.py` has executable top-level code that currently depends on an undefined `model`, `run.py` is the safer module to import from.

---

# Important implementation notes

## This is ternary QAT, not packed 1.58-bit inference

The `BitLinear` layer quantizes its weights to `-1`, `0`, or `1` during the forward pass, but the trainable parameters remain ordinary floating-point PyTorch tensors.

The hosted `model.safetensors` checkpoint is therefore still a floating-point checkpoint, not a physically packed 1.58-bit weight file.

## There is no custom low-bit inference kernel

The layer ultimately calls:

```python
torch.nn.functional.linear(...)
```

with the quantized tensor.

That demonstrates the ternary computation path, but it does not by itself provide the memory footprint or inference-speed benefits of a purpose-built BitNet CUDA/C++/CPU kernel.

## The modification targets GPT-2 MLP projections

`perform_surgery()` replaces modules whose names contain:

```text
mlp.c_fc
mlp.c_proj
```

The attention projection layers are not replaced by this implementation.

---

# Project status

This project is an educational experiment in:

- ternary weights;
- quantization-aware training;
- straight-through estimators;
- dynamic model surgery in PyTorch;
- adapting a pretrained GPT-2 architecture to custom linear layers.

The 125M-parameter model is small by modern LLM standards and should be treated primarily as a learning/research experiment rather than a production language model.

## Reported experiment

The original experiment used WikiText-2 and a two-hour training run on an NVIDIA T4. The project reported substantial recovery in language structure after the initial ternary conversion.

These results are experiment-specific and are not a general benchmark of BitNet implementations.

---

# Future work

Useful next steps for the repository would be:

1. make `model_converter.py` a clean importable module with no top-level checkpoint-loading side effects;
2. make `trainer.py` create/load its own model and define the showcase function;
3. add `requirements.txt`;
4. add CLI arguments for model paths, prompts, generation settings, training duration, batch size, and learning rate;
5. save enough custom architecture metadata for simpler Hugging Face loading;
6. implement packed ternary weights and optimized low-bit kernels if real memory/inference gains are the goal;
7. add a small smoke test that verifies checkpoint loading and text generation.

---

## License / disclaimer

This repository is an educational project intended to demonstrate the basic ideas behind ternary quantization-aware training and custom PyTorch autograd behavior.
