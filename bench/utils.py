# !usr/bin/env python
# -*- coding:utf-8 -*-

"""
Description  :
Version      : 1.0
Author       : MrYXJ
Mail         : yxj2017@gmail.com
Github       : https://github.com/MrYxJ
Date         : 2023-08-19 10:28:55
LastEditTime : 2023-09-07 23:39:17
Copyright (C) 2023 mryxj. All rights reserved.
"""

import torch
import torch.nn as nn

from calflops.calculate_pipline import CalFlopsPipline
from calflops.utils import flops_to_string
from calflops.utils import generate_transformer_input
from calflops.utils import macs_to_string
from calflops.utils import params_to_string


# Adapted from: https://github.com/chengzegang/calculate-flops.pytorch
# NOTE: reture generated results to avoid forward twice.
def calculate_flops(
    model,
    input_shape=None,
    transformer_tokenizer=None,
    args=[],
    kwargs={},
    forward_mode="forward",
    include_backPropagation=False,
    compute_bp_factor=2.0,
    print_results=False,
    print_detailed=False,
    output_as_string=False,
    output_precision=2,
    output_unit=None,
    ignore_modules=None,
    is_sparse=False,
    device=None,
):
    """Returns the total floating-point operations, MACs, and parameters of a model.

    Args:
        model ([torch.nn.Module]): The model of input must be a PyTorch model.
        input_shape (tuple, optional): Input shape to the model. If args and kwargs is empty, the model takes a tensor with this shape as the only positional argument. Default to [].
        transformers_tokenizer (None, optional): Transforemrs Toekenizer must be special if model type is transformers and args、kwargs is empty. Default to None
        args (list, optional): list of positional arguments to the model, such as bert input args is [input_ids, token_type_ids, attention_mask]. Default to []
        kwargs (dict, optional): dictionary of keyword arguments to the model, such as bert input kwargs is {'input_ids': ..., 'token_type_ids':..., 'attention_mask':...}. Default to {}
        forward_mode (str, optional): To determine the mode of model inference, Default to 'forward'. And use 'generate' if model inference uses model.generate().
        include_backPropagation (bool, optional): Decides whether the final return FLOPs computation includes the computation for backpropagation.
        compute_bp_factor (float, optional): The model backpropagation is a multiple of the forward propagation computation. Default to 2.
        print_results (bool, optional): Whether to print the model profile. Defaults to True.
        print_detailed (bool, optional): Whether to print the detailed model profile. Defaults to True.
        output_as_string (bool, optional): Whether to print the output as string. Defaults to True.
        output_precision (int, optional) : Output holds the number of decimal places if output_as_string is True. Default to 2.
        output_unit (str, optional): The unit used to output the result value, such as T, G, M, and K. Default is None, that is the unit of the output decide on value.
        ignore_modules ([type], optional): the list of modules to ignore during profiling. Defaults to None.
        is_sparse (bool, optional): Whether to exclude sparse matrix flops. Defaults to False.

    Example:
    .. code-block:: python
    from calflops import calculate_flops

    # Deep Learning Model, such as alexnet.
    from torchvision import models

    model = models.alexnet()
    batch_size = 1
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=(batch_size, 3, 224, 224),
                                          output_as_string=True,
                                          output_precision=4)
    print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    #Alexnet FLOPs:1.4297 GFLOPS   MACs:714.188 MMACs   Params:61.1008 M

    # Transformers Model, such as bert.
    from transformers import AutoModel
    from transformers import AutoTokenizer
    batch_size = 1
    max_seq_length = 128
    model_name = "hfl/chinese-roberta-wwm-ext/"
    model_save = "../pretrain_models/" + model_name
    model = AutoModel.from_pretrained(model_save)
    tokenizer = AutoTokenizer.from_pretrained(model_save)
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=(batch_size, max_seq_length),
                                          transformer_tokenizer=tokenizer)
    print("Bert(hfl/chinese-roberta-wwm-ext) FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    #Bert(hfl/chinese-roberta-wwm-ext) FLOPs:22.36 GFLOPS   MACs:11.17 GMACs   Params:102.27 M

    # Large Languase Model, such as llama2-7b.
    from transformers import LlamaTokenizer
    from transformers import LlamaForCausalLM
    batch_size = 1
    max_seq_length = 128
    model_name = "llama2_hf_7B"
    model_save = "../model/" + model_name
    model = LlamaForCausalLM.from_pretrained(model_save)
    tokenizer = LlamaTokenizer.from_pretrained(model_save)
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=(batch_size, max_seq_length),
                                          transformer_tokenizer=tokenizer)
    print("Llama2(7B) FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    #Llama2(7B) FLOPs:1.7 TFLOPS   MACs:850.00 GMACs   Params:6.74 B

    Returns:
        The number of floating-point operations, multiply-accumulate operations (MACs), and parameters in the model.
    """

    assert isinstance(model, nn.Module), "model must be a PyTorch module"
    # assert transformers_tokenizer and auto_generate_transformers_input and "transformers" in str(type(model)), "The model must be a transformers model if args of auto_generate_transformers_input is True and transformers_tokenizer is not None"
    model.eval()

    is_transformer = True if "transformers" in str(type(model)) else False

    calculate_flops_pipline = CalFlopsPipline(
        model=model,
        include_backPropagation=include_backPropagation,
        compute_bp_factor=compute_bp_factor,
        is_sparse=is_sparse,
    )
    calculate_flops_pipline.start_flops_calculate(ignore_list=ignore_modules)

    if input_shape is not None:
        assert (
            len(args) == 0 and len(kwargs) == 0
        ), "args and kwargs must be empty value if input_shape is not None, then will be generate random input by inpust_shape"
        assert type(input_shape) is tuple, "input_shape must be a tuple"
        assert (
            len(input_shape) >= 1
        ), "input_shape must have at least one element"

        if transformer_tokenizer is None:  # model is not transformers model
            assert (
                is_transformer is False
            ), "the model is must not transformer model if input_shape is not None and transformer_tokenizer is None"
            try:
                input = torch.ones(()).new_empty(
                    (*input_shape,),
                    dtype=next(model.parameters()).dtype,
                    device=device,
                )
            except StopIteration:
                input = torch.ones(()).new_empty((*input_shape,))
            args = [input]
        else:
            assert (
                len(input_shape) == 2
            ), "the format of input_shape must be (batch_size, seq_len) if model is transformers model and auto_generate_transformers_input if True"
            kwargs = generate_transformer_input(
                input_shape=input_shape,
                model_tokenizer=transformer_tokenizer,
                device=device,
            )
    else:
        assert transformer_tokenizer or (
            len(args) > 0 or len(kwargs) > 0
        ), "input_shape or args or kwargs one of there parameters must specified if auto_generate_input is False"
        if transformer_tokenizer:
            kwargs = generate_transformer_input(
                input_shape=None,
                model_tokenizer=transformer_tokenizer,
                device=device,
            )

    if forward_mode == "forward":
        results = model(*args, **kwargs)
    elif forward_mode == "generate":
        results = model.generate(*args, **kwargs)
    else:
        raise NotImplementedError(
            "forward_mode should be either forward or generate"
        )

    flops = calculate_flops_pipline.get_total_flops()
    macs = calculate_flops_pipline.get_total_macs()
    params = calculate_flops_pipline.get_total_params()

    if print_results:
        return_print = calculate_flops_pipline.print_model_pipline(
            units=output_unit,
            precision=output_precision,
            print_detailed=print_detailed,
        )

    calculate_flops_pipline.end_flops_calculate()

    if include_backPropagation:
        flops = flops * (1 + compute_bp_factor)
        macs = macs * (1 + compute_bp_factor)

    if output_as_string:
        return (
            flops_to_string(
                flops, units=output_unit, precision=output_precision
            ),
            macs_to_string(macs, units=output_unit, precision=output_precision),
            params_to_string(
                params, units=output_unit, precision=output_precision
            ),
        )

    return flops, macs, params, results
