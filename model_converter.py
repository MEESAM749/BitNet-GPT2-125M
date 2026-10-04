"""Utilities for converting GPT-2 MLP projections to ternary BitLinear layers.

This module is intentionally import-safe: importing it does not download a model,
modify a model, or load a checkpoint.  It provides the architecture conversion
and checkpoint-loading primitives used by both ``trainer.py`` and ``run.py``.

The project keeps the original experiment's ternary forward path: trainable
floating-point weights are quantized to {-1, 0, 1} on every forward pass and the
straight-through estimator (STE) passes gradients through the quantizer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
try:
    from transformers.pytorch_utils import Conv1D
except ImportError:  # Transformers may re-export it at the package root.
    from transformers import Conv1D

DEFAULT_TRAINED_REPO = "123aloo123/BitNet-GPT2-125M-Ternary"
TARGET_SUFFIXES = ("mlp.c_fc", "mlp.c_proj")


class BitNetSTE(torch.autograd.Function):
    """Straight-through estimator for ternary weight quantization."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> torch.Tensor:
        gamma = weight.abs().mean()
        return torch.clamp(
            torch.round(weight / (gamma + 1e-5)),
            min=-1,
            max=1,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class BitLinear(nn.Linear):
    """Linear layer whose weights are ternarized during the forward pass."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized_weight = BitNetSTE.apply(self.weight)
        return nn.functional.linear(x, quantized_weight, self.bias)


def _is_target_module(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in TARGET_SUFFIXES)


def _replace_child(model: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    parent_name, child_name = qualified_name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    setattr(parent, child_name, new_module)


def _convert_projection(module: nn.Module, preserve_bias: bool) -> BitLinear:
    """Convert one GPT-2 projection module to BitLinear while preserving weights."""

    if isinstance(module, Conv1D):
        # Hugging Face GPT-2 Conv1D stores weights as [in_features, out_features].
        in_features, out_features = module.weight.shape
        weight = module.weight.t()
    elif isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        weight = module.weight
    else:
        raise TypeError(
            f"Expected GPT-2 Conv1D or nn.Linear for ternary conversion, got "
            f"{type(module).__name__}."
        )

    use_bias = preserve_bias and getattr(module, "bias", None) is not None
    new_layer = BitLinear(in_features, out_features, bias=use_bias)
    new_layer = new_layer.to(device=module.weight.device, dtype=module.weight.dtype)

    with torch.no_grad():
        new_layer.weight.copy_(weight)
        if use_bias:
            new_layer.bias.copy_(module.bias)

    return new_layer


def perform_surgery(
    model: nn.Module,
    *,
    preserve_bias: bool = True,
    verbose: bool = False,
) -> nn.Module:
    """Replace GPT-2 MLP ``c_fc`` and ``c_proj`` projections with BitLinear.

    Parameters
    ----------
    model:
        A GPT-2-family causal language model.
    preserve_bias:
        Copy the original GPT-2 MLP biases into the replacement layers.  New
        training runs should normally leave this enabled.  Legacy checkpoints
        from this project omitted these biases; ``load_bitnet_model`` detects
        that layout automatically.
    verbose:
        Print each converted module name.

    Returns
    -------
    nn.Module
        The same model object, modified in place.
    """

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if _is_target_module(name)
    ]

    if not targets:
        raise ValueError(
            "No GPT-2 MLP projections named 'mlp.c_fc' or 'mlp.c_proj' were found. "
            "This converter currently targets Hugging Face GPT-2 style models."
        )

    converted = 0
    for name, module in targets:
        if isinstance(module, BitLinear):
            continue
        new_layer = _convert_projection(module, preserve_bias=preserve_bias)
        _replace_child(model, name, new_layer)
        converted += 1
        if verbose:
            print(f"Converted {name} -> BitLinear")

    # Save enough metadata for this project's loader to reconstruct the custom
    # architecture later.  AutoModel alone still cannot instantiate BitLinear.
    if hasattr(model, "config"):
        model.config.bitnet_ternary_mlp = True
        model.config.bitnet_mlp_bias = bool(preserve_bias)
        model.config.bitnet_target_modules = list(TARGET_SUFFIXES)

    if verbose:
        print(f"Converted {converted} GPT-2 MLP projections to BitLinear.")

    return model


def convert_pretrained_model(
    base_model: str = "gpt2",
    *,
    preserve_bias: bool = True,
    verbose: bool = False,
):
    """Load a pretrained causal LM and convert its GPT-2 MLPs to BitLinear."""

    model = AutoModelForCausalLM.from_pretrained(base_model)
    perform_surgery(model, preserve_bias=preserve_bias, verbose=verbose)
    return model


def _resolve_weight_file(source: str) -> Tuple[str, str]:
    """Return (format, local_path) for a local directory or Hub repository."""

    path = Path(source).expanduser()
    if path.is_dir():
        safetensors_path = path / "model.safetensors"
        pytorch_path = path / "pytorch_model.bin"
        if safetensors_path.exists():
            return "safetensors", str(safetensors_path)
        if pytorch_path.exists():
            return "pytorch", str(pytorch_path)
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin found in {path}."
        )

    try:
        file_path = hf_hub_download(repo_id=source, filename="model.safetensors")
        return "safetensors", file_path
    except Exception as safetensors_error:
        try:
            file_path = hf_hub_download(repo_id=source, filename="pytorch_model.bin")
            return "pytorch", file_path
        except Exception:
            raise FileNotFoundError(
                f"Could not find model.safetensors or pytorch_model.bin in '{source}'."
            ) from safetensors_error


def _load_raw_state_dict(source: str) -> Dict[str, torch.Tensor]:
    weight_format, file_path = _resolve_weight_file(source)
    if weight_format == "safetensors":
        return dict(load_file(file_path, device="cpu"))

    try:
        state = torch.load(file_path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        state = torch.load(file_path, map_location="cpu")

    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint {file_path} does not contain a state dict.")

    return dict(state)


def _checkpoint_has_mlp_bias(state_dict: Mapping[str, torch.Tensor]) -> bool:
    return any(
        key.endswith("mlp.c_fc.bias") or key.endswith("mlp.c_proj.bias")
        for key in state_dict
    )


def _adapt_state_dict_shapes(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Adapt legacy/standard GPT-2 projection orientation when necessary.

    BitLinear uses nn.Linear's [out, in] layout.  GPT-2 Conv1D uses [in, out].
    Existing project checkpoints already use BitLinear orientation, while a
    standard GPT-2 state dict uses Conv1D orientation.  We only transpose a
    target weight when its transpose exactly matches the expected shape.
    """

    expected = model.state_dict()
    adapted: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        expected_value = expected.get(key)
        if expected_value is None:
            adapted[key] = value
            continue

        if value.shape == expected_value.shape:
            adapted[key] = value
            continue

        if (
            value.ndim == 2
            and _is_target_module(key.removesuffix(".weight"))
            and value.t().shape == expected_value.shape
        ):
            adapted[key] = value.t().contiguous()
            continue

        adapted[key] = value

    return adapted


def load_bitnet_model(
    source: str = DEFAULT_TRAINED_REPO,
    *,
    device: Optional[str | torch.device] = None,
    verbose: bool = False,
):
    """Load a trained checkpoint using the custom BitLinear architecture.

    ``source`` may be a local model directory or a Hugging Face repository ID.
    Legacy checkpoints created by the original experiment are detected by the
    absence of MLP bias tensors and reconstructed with ``bias=False``.
    """

    state_dict = _load_raw_state_dict(source)
    config = AutoConfig.from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained(source)

    # The checkpoint tensors are authoritative.  This keeps the original
    # no-bias checkpoint loadable even though new training runs preserve bias.
    preserve_bias = _checkpoint_has_mlp_bias(state_dict)

    model = AutoModelForCausalLM.from_config(config)
    perform_surgery(model, preserve_bias=preserve_bias, verbose=verbose)

    adapted_state = _adapt_state_dict_shapes(model, state_dict)
    incompatible = model.load_state_dict(adapted_state, strict=False)
    model.tie_weights()

    # A tied lm_head may legitimately be absent from a Safetensors checkpoint.
    ignorable_missing = {"lm_head.weight"}
    meaningful_missing = [
        key for key in incompatible.missing_keys if key not in ignorable_missing
    ]

    if meaningful_missing:
        raise RuntimeError(
            "Checkpoint is missing required model parameters: "
            + ", ".join(meaningful_missing[:20])
        )
    if incompatible.unexpected_keys and verbose:
        print("Unexpected checkpoint keys:", incompatible.unexpected_keys)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(resolved_device)
    model.eval()
    return model, tokenizer, resolved_device


def save_bitnet_model(model, tokenizer, output_dir: str) -> None:
    """Save a converted/trained model and tokenizer in a reloadable directory."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a pretrained GPT-2 model to this project's ternary BitLinear architecture."
    )
    parser.add_argument("--base-model", default="gpt2")
    parser.add_argument("--output-dir", default="./bitnet_gpt2_init")
    parser.add_argument(
        "--drop-mlp-bias",
        action="store_true",
        help="Reproduce the original experiment's bias=False conversion. New runs should normally keep biases.",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    print(f"Loading {args.base_model}...")
    model = convert_pretrained_model(
        args.base_model,
        preserve_bias=not args.drop_mlp_bias,
        verbose=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    save_bitnet_model(model, tokenizer, args.output_dir)
    print(f"Saved QAT-ready ternary model to {args.output_dir}")


if __name__ == "__main__":
    main()
