import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
import os

# ==========================================
# 1. CUSTOM 1.58-BIT ARCHITECTURE
# ==========================================
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

# ==========================================
# 2. LOADING THE LOCAL MODEL
# ==========================================
def load_local_model(local_dir="."):
    print("Loading tokenizer and config from local folder...")
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    config = AutoConfig.from_pretrained(local_dir)
    
    print("Building blank model and performing surgery...")
    model = AutoModelForCausalLM.from_config(config)
    perform_surgery(model)
    
    print("Loading weights from model.safetensors...")
    # Read the raw file
    state_dict = load_file(os.path.join(local_dir, "model.safetensors"))
    
    # We delete the transpose loop! Just shove the raw weights straight in.
    print("Injecting weights into the 1.58-bit brain...")
    model.load_state_dict(state_dict, strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

# ==========================================
# 3. CHAT / GENERATION INTERFACE
# ==========================================
if __name__ == "__main__":
    # We pass "." to tell it to look in the current folder
    model, tokenizer, device = load_local_model(".")
    print("\nModel loaded successfully! Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        prompt = input("\nUser Prompt: ")
        if prompt.lower() in ['quit', 'exit']:
            break
            
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=60, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nBitNet: {response}")