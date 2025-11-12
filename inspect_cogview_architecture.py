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
    # Get model path from environment variable
    model_path = os.environ.get("MODEL_DIR")
    if not model_path:
        print("Error: Please set MODEL_DIR environment variable")
        print("Example: export MODEL_DIR=zai-org/CogView3-Plus-3B")
        return

    # Infer model type from path
    model_path_lower = model_path.lower()
    if "cogview3" in model_path_lower or "3-plus" in model_path_lower:
        model_name = "CogView3-Plus-3B"
        pipeline_class = "CogView3PlusPipeline"
    elif "cogview4" in model_path_lower or "4-6b" in model_path_lower:
        model_name = "CogView4-6B"
        pipeline_class = "CogView4Pipeline"
    else:
        print(
            f"Warning: Could not determine model type from path: {model_path}"
        )
        print("Please ensure path contains 'cogview3' or 'cogview4'")
        return

    print(f"Detected model: {model_name}")
    print(f"Pipeline class: {pipeline_class}")
    print(f"Loading from: {model_path}")

    try:
        # Import the appropriate pipeline
        if pipeline_class == "CogView3PlusPipeline":
            from diffusers import CogView3PlusPipeline

            pipe = CogView3PlusPipeline.from_pretrained(
                model_path, torch_dtype=torch.bfloat16
            )
        else:  # CogView4Pipeline
            from diffusers import CogView4Pipeline

            pipe = CogView4Pipeline.from_pretrained(
                model_path, torch_dtype=torch.bfloat16
            )

        print_linear_layers(pipe.transformer, model_name)

    except Exception as e:
        print(f"Failed to load {model_name}: {e}")


if __name__ == "__main__":
    main()
