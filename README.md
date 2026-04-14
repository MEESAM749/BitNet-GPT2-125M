# BitNet-GPT2-125M: 1.58-bit Quantization-Aware Training

This repository demonstrates the end-to-end conversion, Quantization-Aware Training (QAT), and inference of a 1.58-bit Large Language Model built entirely from scratch. By performing "Model Surgery" on a standard Hugging Face GPT-2 (125M), this project replaces standard 16-bit floating-point linear layers with custom **Ternary (-1, 0, 1) BitLinear layers**, inspired by Microsoft's *The Era of 1-bit LLMs* research.

## The Core Concept: 1.58-bit Inference

Standard LLMs rely on highly precise decimal weights (FP16 or BF16) which require expensive matrix multiplications. This project implements a **BitLinear** architecture that restricts weights to ternary values `{-1, 0, 1}`.

This transition mathematically simplifies inference from complex matrix multiplications to highly efficient, hardware-friendly addition and subtraction, dramatically lowering memory footprint and increasing inference speed.

## Key Features

* **Custom `BitLinear` PyTorch Module:** A drop-in replacement for `nn.Linear` featuring dynamic AbsMean scaling.
* **Straight-Through Estimator (STE):** A custom `torch.autograd.Function` that allows backpropagation to bypass the non-differentiable step function of the ternary quantizer, updating hidden "shadow" FP16 weights during training.
* **Live Architecture Surgery:** A dynamic pipeline that walks the model graph, replacing standard `Conv1D` / `Linear` layers with `BitLinear` modules while copying initial shadow weights.
* **Memory-Safe QAT Loop:** Implements micro-batching and gradient accumulation to perform full Quantization-Aware Training within the 15GB VRAM constraints of standard consumer GPUs (e.g., NVIDIA T4).

## Architecture: The `BitNetSTE`

The magic of this model relies on deceiving the PyTorch computational graph during the backward pass.

1. **Forward Pass:** The weights are scaled by their absolute mean ($\gamma$) and clamped to `[-1, 0, 1]`.
2. **Backward Pass:** The gradient passes *straight through* the quantization function unaltered, allowing the optimizer to adjust the underlying high-precision weights.

```python
class BitNetSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W):
        gamma = W.abs().mean()
        W_quant = torch.clamp(torch.round(W / (gamma + 1e-5)), min=-1, max=1)
        return W_quant

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output # The STE "Lie"
```

## Getting Started

### Prerequisites

```bash
pip install torch transformers datasets
```

### 1. Perform Model Surgery

Load a standard GPT-2 model and inject the `BitLinear` layers:

```python
from transformers import AutoModelForCausalLM
from custom_architecture import perform_surgery

model = AutoModelForCausalLM.from_pretrained("gpt2")
perform_surgery(model)
```

### 2. Fine-Tuning

Because the initial transition to ternary weights causes massive precision loss (acting like a lobotomy to the pre-trained model), it must undergo Quantization-Aware Training to recover its reasoning capabilities.

Run the training loop on the WikiText dataset:

```bash
python train_bitnet.py
```

*Note: The training script includes an automatic evolution showcase that generates text every 15 minutes, allowing you to monitor the model's recovery of grammar and syntax in real-time.*

## Results & Evolution

During a standard 2-hour training run on an NVIDIA T4, the model successfully demonstrated the ability to recover from the initial quantization shock.

* **Epoch 1 Average Loss:** `8.6477` (Random/Confused)
* **Epoch 3 Average Loss:** `5.9542` (Grammar recovery)

**Sample Generation (Post-Surgery, Pre-Training):**

> *Prompt:* The future of artificial intelligence is
> *Output:* `zxq wlp rtb the a of to in...`

**Sample Generation (After 2 Hours of QAT):**

> *Prompt:* The future of artificial intelligence is
> *Output:* `the most powerful and influential than any of the most important computer, and the computer is the best known for its...`

## NOTE: 
# THIS WAS AN EXPERIMENT FOR LEARNING PURPOSES, NOTHING MORE. 125M PARAMS ARE FAR TOO FEW. I WOULD'VE USED A LARGER MODEL BUT I DIDN'T HAVE ENOUGH HW RESOURCES TO TRAIN A LARGER MODEL.

## Future Work

* Integrate **Sub-Layer Normalization (SubLN)** to stabilize activation variance in deeper models.
* Scale up the architecture surgery to a 32B parameter Distilled Teacher/Student framework.
* Implement custom CUDA/C++ kernels (`bitnet.cpp` integration) for actual on-device CPU inference speedups.

*Disclaimer: This is an educational project built to demonstrate the foundational concepts of Quantization-Aware Training and custom PyTorch autograd functions.*
```
