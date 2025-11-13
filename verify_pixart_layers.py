#!/usr/bin/env python3

import os
import torch
from diffusers import PixArtAlphaPipeline, PixArtSigmaPipeline


def load_pixart_model(model_path):
    """Load PixArt pipeline and return transformer."""
    if "Sigma" in model_path:
        pipe = PixArtSigmaPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        )
    else:
        pipe = PixArtAlphaPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        )
    return pipe.transformer


def print_linear_layers(transformer):
    """Print linear layer information from all transformer blocks."""
    total_blocks = len(transformer.transformer_blocks)

    for i in range(total_blocks):
        block = transformer.transformer_blocks[i]
        for name, module in block.named_modules():
            if isinstance(module, torch.nn.Linear):
                print(f"{i}: {name}: {module}")


def main():
    model_path = os.environ.get("PIXART_MODEL_PATH")
    transformer = load_pixart_model(model_path)
    print_linear_layers(transformer)


if __name__ == "__main__":
    main()
