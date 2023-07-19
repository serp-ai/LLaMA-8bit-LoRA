import transformers
import torch
import torch.nn as nn
from typing import Dict


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """From: https://github.com/artidoro/qlora/blob/main/qlora.py
    Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if num_new_tokens > 0:
        input_embeddings_data = input_embeddings.weight.data
        output_embeddings_data = output_embeddings.weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg
        model.tie_weights()

    # Temporary bug fix #214: 
    # freeze embeddings otherwise need to store them with checkpoint
    input_embeddings.weight.requires_grad = False
    output_embeddings.weight.requires_grad = False
    # re-register forward hook
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


def str_or_bool(value):
    if str(value).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str(value).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return str(value)  # if it's not a recognizable boolean, treat it as a string
