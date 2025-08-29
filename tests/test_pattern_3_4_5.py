import dataclasses
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from typing import Tuple, Union
import cache_dit
from cache_dit import ForwardPattern, BlockAdapter


class RandTransformerBlock_Pattern_3(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = torch.randn_like(hidden_states)
        return hidden_states


class RandTransformerBlock_Pattern_4(torch.nn.Module):
    def forward(
        self, hidden_states: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = torch.randn_like(hidden_states)
        encoder_hidden_states = torch.randn_like(hidden_states)
        return hidden_states, encoder_hidden_states


class RandTransformerBlock_Pattern_5(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = torch.randn_like(hidden_states)
        encoder_hidden_states = torch.randn_like(hidden_states)
        return encoder_hidden_states, hidden_states


@dataclasses
class RandTransformer2DModelOutput:
    sample: torch.Tensor = None


class RandTransformer2DModel(torch.nn.Module):
    def __init__(
        self,
        num_blocks: int = 64,
        pattern: ForwardPattern = ForwardPattern.Pattern_3,
    ):
        self.pattern = pattern
        if pattern == ForwardPattern.Pattern_3:
            self.transformer_blocks = nn.ModuleList(
                [RandTransformerBlock_Pattern_3() for _ in range(num_blocks)]
            )
        elif pattern == ForwardPattern.Pattern_4:
            self.transformer_blocks = nn.ModuleList(
                [RandTransformerBlock_Pattern_4() for _ in range(num_blocks)]
            )
        elif pattern == ForwardPattern.Pattern_5:
            self.transformer_blocks = nn.ModuleList(
                [RandTransformerBlock_Pattern_5() for _ in range(num_blocks)]
            )
        else:
            raise ValueError(f"{pattern} is not supported!")

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> Union[torch.Tensor, RandTransformer2DModelOutput]:

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, *args, **kwargs)
            if not isinstance(hidden_states, torch.Tensor):
                if self.pattern.Return_H_First:
                    hidden_states, encoder_hidden_states = hidden_states
                else:
                    encoder_hidden_states, hidden_states = hidden_states

        if encoder_hidden_states is not None:
            assert isinstance(encoder_hidden_states, torch.Tensor)
            encoder_hidden_states = encoder_hidden_states.contiguous()

        output = hidden_states + encoder_hidden_states

        if kwargs.get("return_dict", True):
            return RandTransformer2DModelOutput(sample=output)

        return output


RandPipelineOutput = RandTransformer2DModelOutput


class RandPipeline:

    def __init__(
        self,
        pattern: ForwardPattern = ForwardPattern.Pattern_3,
    ):
        self.transformer = RandTransformer2DModel(pattern=pattern)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        num_inference_steps: int = 50,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, RandPipelineOutput]:
        timesteps = list(range(num_inference_steps))[::-1]
        for i, t in tqdm(enumerate(timesteps)):

            noise_pred = self.transformer(
                hidden_states=hidden_states,
                return_dict=False,
            )

        if not return_dict:
            return (noise_pred,)

        return RandPipelineOutput(sample=noise_pred)


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=int, choices=[3, 4, 5], default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
    if args.pattern == 3:
        pipe = RandPipeline(pattern=ForwardPattern.Pattern_3)
    elif args.pattern == 4:
        pipe = RandPipeline(pattern=ForwardPattern.Pattern_4)
    else:
        pipe = RandPipeline(pattern=ForwardPattern.Pattern_5)

    cache_dit.enable_cache(
        BlockAdapter(
            pipe=pipe,
            transformer=pipe.transformer,
            blocks=pipe.transformer.transformer_blocks,
            blocks_name="transformer_blocks",
        ),
        forward_pattern=pipe.transformer.pattern,
        Fn_compute_blocks=1,
        Bn_compute_blocks=0,
        residual_diff_threshold=0.03,
    )
    bs, seq_len, headdim = 1, 4096, 2048
    hidden_states = torch.randn((bs, seq_len, headdim), dtype=torch.bfloat16)

    output = pipe(hidden_states, num_inference_steps=50)

    cache_dit.summary(pipe, details=True)
