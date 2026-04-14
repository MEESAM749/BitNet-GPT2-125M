from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch
import torch.optim as optim
import gc
import time
import os

# --- 1. SETUP & DATA PREP ---
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

# ---  THE 2-HOUR TIME-BOUND LOOP ---
max_duration_seconds = 2 * 60 * 60  # 2 Hours
start_time = time.time()

# Timers for our showcase and save features
last_showcase_time = start_time
last_save_time = start_time
showcase_interval = 15 * 60  # Every 15 minutes
save_interval = 30 * 60      # Every 30 minutes

checkpoint_dir = "./bitnet_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"\nStarting 2-Hour 1.58-bit Fine-Tuning Run...")
model.train()

step = 0
running_loss = 0.0

# We use a 'while' loop based on time instead of a 'for' loop based on epochs, you can change this if you want.
while (time.time() - start_time) < max_duration_seconds:

    for i in range(0, len(texts), batch_size):
        # Check time before every batch
        current_time = time.time()
        if (current_time - start_time) >= max_duration_seconds:
            break # 2 hours are up!

        # --- EVENT: 15-Minute Showcase ---
        if (current_time - last_showcase_time) >= showcase_interval:
            showcase_evolution(step, (current_time - start_time) / 60)
            last_showcase_time = current_time

        # --- EVENT: 30-Minute Checkpoint Save ---
        if (current_time - last_save_time) >= save_interval:
            save_path = f"{checkpoint_dir}/model_min_{int((current_time - start_time)/60)}"
            model.save_pretrained(save_path)
            print(f" Checkpoint saved to {save_path}")
            last_save_time = current_time

        # --- STANDARD TRAINING ---
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

        # Print a simple progress log every 50 steps
        if step % 50 == 0:
            avg_loss = running_loss / 50
            elapsed_mins = (time.time() - start_time) / 60
            print(f"Step {step} | Time: {elapsed_mins:.1f}m / 120.0m | Avg Loss: {avg_loss:.4f}")
            running_loss = 0.0

print("\n2-Hour Training Complete! Saving final model...")
model.save_pretrained("./bitnet_final_2hr")
tokenizer.save_pretrained("./bitnet_final_2hr")
print("Final model saved to './bitnet_final_2hr'")