import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

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

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("gpt2")
perform_surgery(model)

print("\nVerification:")
print(model.transformer.h[0].mlp.c_fc)

#!pip install datasets

from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch
import torch.optim as optim
import gc
import time
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Downloading full training data (WikiText)...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")


print("Filtering dataset...")
texts = [text for text in dataset['text'] if len(text.strip()) > 50]
print(f"Total sentences ready for training: {len(texts)}")

optimizer = optim.AdamW(model.parameters(), lr=5e-5)
batch_size = 4

def showcase_evolution(step, elapsed_time):
    """Pauses training to let the model generate text so we can watch it learn."""
    model.eval() # Turn off training mode
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"\n[{elapsed_time:.1f} Minutes] - Model Evolution Showcase (Step {step}):")
    print("-" * 50)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=35, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.7)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt:   {prompt}")
    print(f"Response: {generated_text.replace(prompt, '').strip()}")
    print("-" * 50 + "\n")

    model.train()


    del inputs, outputs
    torch.cuda.empty_cache()


max_duration_seconds = 2 * 60 * 60  # 2 Hours
start_time = time.time()


last_showcase_time = start_time
last_save_time = start_time
showcase_interval = 15 * 60 
save_interval = 30 * 60      

checkpoint_dir = "./bitnet_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"\nStarting 2-Hour 1.58-bit Fine-Tuning Run...")
model.train()

step = 0
running_loss = 0.0


while (time.time() - start_time) < max_duration_seconds:

    for i in range(0, len(texts), batch_size):

        current_time = time.time()
        if (current_time - start_time) >= max_duration_seconds:
            break 


        if (current_time - last_showcase_time) >= showcase_interval:
            showcase_evolution(step, (current_time - start_time) / 60)
            last_showcase_time = current_time

        if (current_time - last_save_time) >= save_interval:
            save_path = f"{checkpoint_dir}/model_min_{int((current_time - start_time)/60)}"
            model.save_pretrained(save_path)
            print(f" Checkpoint saved to {save_path}")
            last_save_time = current_time


        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        optimizer.zero_grad()

        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['input_ids'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        step += 1

        # Memory cleanup
        del inputs, outputs, loss
        torch.cuda.empty_cache()
        gc.collect()

        if step % 50 == 0:
            avg_loss = running_loss / 50
            elapsed_mins = (time.time() - start_time) / 60
            print(f"Step {step} | Time: {elapsed_mins:.1f}m / 120.0m | Avg Loss: {avg_loss:.4f}")
            running_loss = 0.0

print("\n 2-Hour Training Complete! Saving final model...")
model.save_pretrained("./bitnet_final_2hr")
tokenizer.save_pretrained("./bitnet_final_2hr")
print(" Final model saved to './bitnet_final_2hr'")

model.eval()


prompt_text = "The future of artificial intelligence is"
inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

print(f"Prompt: {prompt_text}")
print("-" * 40)
print("Generating response...\n")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=30,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,     
        temperature=0.7      
    )


generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)