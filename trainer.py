"""Time-bounded QAT trainer for the BitNet-GPT2-125M experiment."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, set_seed

from model_converter import convert_pretrained_model, save_bitnet_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert GPT-2 MLP projections to ternary BitLinear layers and fine-tune "
            "them with quantization-aware training on WikiText-2."
        )
    )
    parser.add_argument("--base-model", default="gpt2")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--output-dir", default="./bitnet_final_2hr")
    parser.add_argument("--checkpoint-dir", default="./bitnet_checkpoints")
    parser.add_argument("--duration-hours", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--showcase-minutes", type=float, default=15.0)
    parser.add_argument("--save-minutes", type=float, default=30.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--min-text-chars", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt",
        default="The future of artificial intelligence is",
        help="Prompt used for periodic qualitative generation.",
    )
    parser.add_argument(
        "--legacy-drop-mlp-bias",
        action="store_true",
        help=(
            "Reproduce the original experiment exactly by dropping GPT-2 MLP biases. "
            "By default this fixed trainer preserves them."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.duration_hours <= 0:
        raise ValueError("--duration-hours must be greater than zero")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be greater than zero")
    if args.max_length <= 1:
        raise ValueError("--max-length must be greater than one")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than zero")
    if args.log_every <= 0:
        raise ValueError("--log-every must be greater than zero")


def load_training_texts(args: argparse.Namespace) -> List[str]:
    print(
        f"Loading {args.dataset}/{args.dataset_config} ({args.dataset_split})..."
    )
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
    )

    if "text" not in dataset.column_names:
        raise ValueError(
            f"Dataset must contain a 'text' column; found {dataset.column_names}."
        )

    texts = [
        text
        for text in dataset["text"]
        if isinstance(text, str) and len(text.strip()) > args.min_text_chars
    ]
    if not texts:
        raise ValueError("No training examples remained after text filtering.")

    print(f"Training examples after filtering: {len(texts):,}")
    return texts


def showcase_evolution(model, tokenizer, device, prompt: str, step: int, elapsed_minutes: float) -> None:
    """Generate a short sample without disturbing the model's training mode."""

    was_training = model.training
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[1]

    print(f"\n[{elapsed_minutes:.1f} min] Showcase at optimizer step {step}")
    print("-" * 60)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    continuation = tokenizer.decode(
        outputs[0, prompt_tokens:],
        skip_special_tokens=True,
    )
    print(f"Prompt:   {prompt}")
    print(f"Response: {continuation.strip()}")
    print("-" * 60)

    if was_training:
        model.train()


def save_checkpoint(model, tokenizer, output_dir: str, step: int, elapsed_minutes: float) -> None:
    save_bitnet_model(model, tokenizer, output_dir)
    state_path = Path(output_dir) / "training_state.pt"
    torch.save(
        {
            "optimizer_step": step,
            "elapsed_minutes": elapsed_minutes,
        },
        state_path,
    )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading pretrained base model: {args.base_model}")
    model = convert_pretrained_model(
        args.base_model,
        preserve_bias=not args.legacy_drop_mlp_bias,
        verbose=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    texts = load_training_texts(args)
    loader = DataLoader(texts, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    max_duration = args.duration_hours * 60 * 60
    showcase_interval = args.showcase_minutes * 60
    save_interval = args.save_minutes * 60
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()
    last_showcase = start_time
    last_save = start_time
    optimizer_step = 0
    micro_step = 0
    loss_sum = 0.0
    loss_count = 0
    stop_requested = False

    optimizer.zero_grad(set_to_none=True)
    model.train()

    print(
        f"\nStarting ternary QAT for up to {args.duration_hours:.2f} hours "
        f"(batch={args.batch_size}, grad_accum={args.gradient_accumulation_steps})."
    )

    try:
        while not stop_requested:
            for batch_texts in loader:
                now = time.monotonic()
                elapsed = now - start_time
                if elapsed >= max_duration:
                    stop_requested = True
                    break

                if args.showcase_minutes > 0 and now - last_showcase >= showcase_interval:
                    showcase_evolution(
                        model,
                        tokenizer,
                        device,
                        args.prompt,
                        optimizer_step,
                        elapsed / 60,
                    )
                    last_showcase = time.monotonic()

                if args.save_minutes > 0 and now - last_save >= save_interval:
                    minutes = int(elapsed / 60)
                    save_path = checkpoint_dir / f"model_min_{minutes}"
                    save_checkpoint(
                        model,
                        tokenizer,
                        str(save_path),
                        optimizer_step,
                        elapsed / 60,
                    )
                    print(f"Checkpoint saved to {save_path}")
                    last_save = time.monotonic()

                encoded = tokenizer(
                    list(batch_texts),
                    return_tensors="pt",
                    max_length=args.max_length,
                    truncation=True,
                    padding=True,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}

                # GPT-2 shifts labels internally. Padding must be ignored rather
                # than trained as repeated EOS tokens.
                labels = encoded["input_ids"].clone()
                labels[encoded["attention_mask"] == 0] = -100

                outputs = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    labels=labels,
                )
                raw_loss = outputs.loss
                (raw_loss / args.gradient_accumulation_steps).backward()

                loss_sum += raw_loss.detach().item()
                loss_count += 1
                micro_step += 1

                if micro_step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

                    if optimizer_step % args.log_every == 0:
                        elapsed_minutes = (time.monotonic() - start_time) / 60
                        average_loss = loss_sum / max(loss_count, 1)
                        print(
                            f"Step {optimizer_step} | "
                            f"Time {elapsed_minutes:.1f}m / {args.duration_hours * 60:.1f}m | "
                            f"Avg loss {average_loss:.4f}"
                        )
                        loss_sum = 0.0
                        loss_count = 0

                del encoded, labels, outputs, raw_loss

            # DataLoader is finite; if time remains, start another shuffled epoch.
            if time.monotonic() - start_time >= max_duration:
                stop_requested = True

    except KeyboardInterrupt:
        print("\nTraining interrupted by user; saving current model.")

    # Apply a final partially accumulated gradient instead of silently discarding it.
    if micro_step % args.gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1

    elapsed_minutes = (time.monotonic() - start_time) / 60
    print(f"\nTraining finished after {elapsed_minutes:.1f} minutes and {optimizer_step} optimizer steps.")
    save_checkpoint(
        model,
        tokenizer,
        args.output_dir,
        optimizer_step,
        elapsed_minutes,
    )
    print(f"Final model saved to {args.output_dir}")

    showcase_evolution(
        model,
        tokenizer,
        device,
        args.prompt,
        optimizer_step,
        elapsed_minutes,
    )


if __name__ == "__main__":
    main()
