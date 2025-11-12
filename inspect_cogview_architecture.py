#!/usr/bin/env python3
import os
import torch
import torch.nn as nn


def print_linear_layers(model, model_name):
    print(f"\n=== {model_name} ===")

    def traverse(module, prefix=""):
        for name, layer in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
                param_count = (
                    layer.weight.numel() + layer.bias.numel()
                    if layer.bias is not None
                    else layer.weight.numel()
                )
                print(
                    f"{full_name}: {in_features} -> {out_features} ({param_count:,} params)"
                )
            else:
                traverse(layer, full_name)

    traverse(model)


def main():
    # CogView3-Plus-3B
    cogview3_path = os.environ.get("COGVIEW3_DIR", "zai-org/CogView3-Plus-3B")
    try:
        from diffusers import CogView3PlusPipeline

        print(f"Loading CogView3-Plus-3B from: {cogview3_path}")
        pipe3 = CogView3PlusPipeline.from_pretrained(
            cogview3_path, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        print_linear_layers(pipe3.transformer, "CogView3-Plus-3B")
    except Exception as e:
        print(f"Failed to load CogView3-Plus-3B: {e}")

    # CogView4-6B
    cogview4_path = os.environ.get("COGVIEW4_DIR", "zai-org/CogView4-6B")
    try:
        from diffusers import CogView4Pipeline

        print(f"Loading CogView4-6B from: {cogview4_path}")
        pipe4 = CogView4Pipeline.from_pretrained(
            cogview4_path, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        print_linear_layers(pipe4.transformer, "CogView4-6B")
    except Exception as e:
        print(f"Failed to load CogView4-6B: {e}")


if __name__ == "__main__":
    main()
