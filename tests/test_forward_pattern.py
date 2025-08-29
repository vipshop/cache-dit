import dataclasses
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from typing import Tuple, Union

import cache_dit
from cache_dit import ForwardPattern, BlockAdapter


RATIO = 0.7
RAND_RATIO = 0.5


class RandTransformerBlock_Pattern_0(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = torch.FloatTensor([RATIO])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < RAND_RATIO:
            hidden_states = torch.randn_like(hidden_states)
            encoder_hidden_states = torch.randn_like(encoder_hidden_states)
        else:
            hidden_states = hidden_states * self.r.to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
            encoder_hidden_states = encoder_hidden_states * self.r.to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
        return hidden_states, encoder_hidden_states


class RandTransformerBlock_Pattern_1(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = torch.FloatTensor([RATIO])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < RAND_RATIO:
            hidden_states = torch.randn_like(hidden_states)
            encoder_hidden_states = torch.randn_like(encoder_hidden_states)
        else:
            hidden_states = hidden_states * self.r.to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
            encoder_hidden_states = encoder_hidden_states * self.r.to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
        return encoder_hidden_states, hidden_states


class RandTransformerBlock_Pattern_2(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = torch.FloatTensor([RATIO])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < RAND_RATIO:
            hidden_states = torch.randn_like(hidden_states)
            encoder_hidden_states = torch.randn_like(encoder_hidden_states)
        else:
            hidden_states = hidden_states * self.r.to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
            encoder_hidden_states = encoder_hidden_states * self.r.to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
        return hidden_states


class RandTransformerBlock_Pattern_3(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = torch.FloatTensor([RATIO])

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if torch.rand(1).item() < RAND_RATIO:
            hidden_states = torch.randn_like(hidden_states)
        else:
            hidden_states = hidden_states * self.r.to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
        return hidden_states


class RandTransformerBlock_Pattern_4(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = torch.FloatTensor([RATIO])

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < RAND_RATIO:
            hidden_states = torch.randn_like(hidden_states)
            encoder_hidden_states = torch.randn_like(hidden_states)
        else:
            hidden_states = hidden_states * self.r.to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
            encoder_hidden_states = hidden_states
        return hidden_states, encoder_hidden_states


class RandTransformerBlock_Pattern_5(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = torch.FloatTensor([RATIO])

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < RAND_RATIO:
            hidden_states = torch.randn_like(hidden_states)
            encoder_hidden_states = torch.randn_like(hidden_states)
        else:
            hidden_states = hidden_states * self.r.to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
            encoder_hidden_states = hidden_states
        return encoder_hidden_states, hidden_states


@dataclasses.dataclass
class RandTransformer2DModelOutput:
    sample: torch.Tensor = None


class RandTransformer2DModel_Pattern_0_1_2(torch.nn.Module):

    def __init__(
        self,
        num_blocks: int = 64,
        pattern: ForwardPattern = ForwardPattern.Pattern_0,
    ):
        super().__init__()
        self.pattern = pattern
        if pattern == ForwardPattern.Pattern_0:
            self.transformer_blocks = nn.ModuleList(
                [RandTransformerBlock_Pattern_0() for _ in range(num_blocks)]
            )
        elif pattern == ForwardPattern.Pattern_1:
            self.transformer_blocks = nn.ModuleList(
                [RandTransformerBlock_Pattern_1() for _ in range(num_blocks)]
            )
        elif pattern == ForwardPattern.Pattern_2:
            self.transformer_blocks = nn.ModuleList(
                [RandTransformerBlock_Pattern_2() for _ in range(num_blocks)]
            )
        else:
            raise ValueError(f"{pattern} is not supported!")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
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
        else:
            output = hidden_states

        if kwargs.get("return_dict", True):
            return RandTransformer2DModelOutput(sample=output)

        return output


class RandTransformer2DModel_Pattern_3_4_5(torch.nn.Module):

    def __init__(
        self,
        num_blocks: int = 64,
        pattern: ForwardPattern = ForwardPattern.Pattern_3,
    ):
        super().__init__()
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

        encoder_hidden_states = None
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
        else:
            output = hidden_states

        if kwargs.get("return_dict", True):
            return RandTransformer2DModelOutput(sample=output)

        return output


RandPipelineOutput = RandTransformer2DModelOutput


class RandPipeline:

    def __init__(
        self,
        pattern: ForwardPattern = ForwardPattern.Pattern_0,
    ):
        self.pattern = pattern
        if pattern in [
            ForwardPattern.Pattern_0,
            ForwardPattern.Pattern_1,
            ForwardPattern.Pattern_2,
        ]:
            self.is_pattern_0_1_2 = True
            self.transformer = RandTransformer2DModel_Pattern_0_1_2(
                pattern=pattern
            )
        else:
            self.is_pattern_0_1_2 = False
            self.transformer = RandTransformer2DModel_Pattern_3_4_5(
                pattern=pattern
            )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        num_inference_steps: int = 50,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, RandPipelineOutput]:
        timesteps = list(range(num_inference_steps))[::-1]
        for i, t in tqdm(enumerate(timesteps), total=num_inference_steps):

            if not self.is_pattern_0_1_2:
                noise_pred = self.transformer(
                    hidden_states=hidden_states,
                    return_dict=False,
                )
            else:
                noise_pred = self.transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )

        if not return_dict:
            return (noise_pred,)

        return RandPipelineOutput(sample=noise_pred)

    def to(self, *args, **kwargs):
        self.transformer.to(*args, **kwargs)


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern", type=int, choices=[0, 1, 2, 3, 4, 5], default=3
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
    if args.pattern == 0:
        pipe = RandPipeline(pattern=ForwardPattern.Pattern_0)
    elif args.pattern == 1:
        pipe = RandPipeline(pattern=ForwardPattern.Pattern_1)
    elif args.pattern == 2:
        pipe = RandPipeline(pattern=ForwardPattern.Pattern_2)
    elif args.pattern == 3:
        pipe = RandPipeline(pattern=ForwardPattern.Pattern_3)
    elif args.pattern == 4:
        pipe = RandPipeline(pattern=ForwardPattern.Pattern_4)
    else:
        pipe = RandPipeline(pattern=ForwardPattern.Pattern_5)

    pipe.to("cuda")

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
        residual_diff_threshold=0.05,
    )
    bs, seq_len, headdim = 1, 1024, 1024

    hidden_states = torch.normal(
        mean=100.0,
        std=20.0,
        size=(bs, seq_len, headdim),
        dtype=torch.bfloat16,
    ).to("cuda")

    encoder_hidden_states = torch.normal(
        mean=100.0,
        std=20.0,
        size=(bs, seq_len, headdim),
        dtype=torch.bfloat16,
    ).to("cuda")

    if args.pattern in [0, 1, 2]:
        output = pipe(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            num_inference_steps=50,
        )
    else:
        output = pipe(
            hidden_states,
            num_inference_steps=50,
        )

    cache_dit.summary(pipe, details=True)
