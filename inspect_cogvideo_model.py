#!/usr/bin/env python3
"""
Script to inspect CogVideoX1.5-5B transformer for tensor parallelism.

Usage:
    export MODEL_PATH="zai-org/CogVideoX1.5-5B"
    python inspect_cogvideo_model.py
"""

import os
import sys

import torch
from diffusers import CogVideoXTransformer3DModel


def main():
    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        sys.exit(1)

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.float16
    )

    for block_idx, block in enumerate(transformer.transformer_blocks):
        for name, module in block.named_modules():
            if isinstance(module, torch.nn.Linear):
                print(
                    f"block{block_idx}.{name}: Linear({module.in_features} -> {module.out_features})"
                )
