"""Interactive inference for the trained ternary GPT-2 checkpoint."""

from __future__ import annotations

import argparse

import torch

from model_converter import DEFAULT_TRAINED_REPO, load_bitnet_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the trained ternary GPT-2 model.")
    parser.add_argument(
        "--model",
        default=DEFAULT_TRAINED_REPO,
        help="Hugging Face repo ID or local checkpoint directory.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--device",
        default=None,
        help="Optional PyTorch device such as 'cpu', 'cuda', or 'cuda:0'.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model, tokenizer, device = load_bitnet_model(
        args.model,
        device=args.device,
        verbose=True,
    )

    print(f"\nModel loaded on {device}. Type 'quit' or 'exit' to stop.")
    print("-" * 60)

    while True:
        try:
            prompt = input("\nUser Prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt.lower() in {"quit", "exit"}:
            break
        if not prompt:
            continue

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

        continuation = tokenizer.decode(
            outputs[0, prompt_tokens:],
            skip_special_tokens=True,
        )
        print(f"\nBitNet: {continuation.strip()}")


if __name__ == "__main__":
    main()
