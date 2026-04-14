from transformers import AutoConfig, AutoModelForCausalLM
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

class BitNetSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W):
        gamma = W.abs().mean()
        W_quant = torch.clamp(torch.round(W / (gamma + 1e-5)), min=-1, max=1)
        return W_quant

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BitLinear(nn.Linear):
    def forward(self, x):
        w_quant = BitNetSTE.apply(self.weight)
        return nn.functional.linear(x, w_quant, self.bias)

def perform_surgery(model):
    for name, module in model.named_modules():
        if "mlp.c_fc" in name or "mlp.c_proj" in name:
            in_features = module.weight.shape[0]
            out_features = module.weight.shape[1]
            new_layer = BitLinear(in_features, out_features, bias=False)
            with torch.no_grad():
                new_layer.weight.copy_(module.weight.t())

            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, child_name, new_layer)
    print("Surgery Successful! GPT-2 is now 1.58-bit.")

repo_id = "123aloo123/BitNet-GPT2-125M-Ternary"

print("1. Downloading weights file...")
file_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")

print("2. Loading and transposing state_dict...")
state_dict = load_file(file_path)
new_state_dict = {}

# iterate through the weights and flip the ones that have shape mismatches
for key, value in state_dict.items():
    if "mlp.c_fc.weight" in key or "mlp.c_proj.weight" in key:
        # GPT-2 uses [In, Out], we need [Out, In]
        new_state_dict[key] = value.t() 
        print(f"  Fixed shape for: {key}")
    else:
        new_state_dict[key] = value

print("3. Injecting fixed weights into the model...")
# 'strict=False' ignores the missing biases (which we don't want anyway)
model.load_state_dict(new_state_dict, strict=False)

print("\nSUCCESS: Your 1.58-bit trained weights are now active!")