# Code used from https://github.com/CarperAI/trlx/blob/main/examples/hh/sft_hh.py https://github.com/lvwerra/trl/blob/main/examples/sentiment/scripts/gpt-neox-20b_peft/gpt-neo-20b_sentiment_peft.py

from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union

import os
import torch
import torch.nn as nn
import transformers
import accelerate
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, LlamaTokenizer, LlamaForCausalLM, LlamaConfig

from accelerate import init_empty_weights, infer_auto_device_map
from transformers import AutoConfig, set_seed, BitsAndBytesConfig

from utils import CastOutputToFloat, smart_tokenizer_and_embedding_resize, print_trainable_parameters, str_or_bool

import torch.backends.cuda
torch.backends.cuda.matmul.allow_tf32 = True
# Uncomment the following line to enable flash attention (model source code must be modified)
# torch.backends.cuda.enable_flash_sdp(enabled=True)


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None
    )
    r: Optional[int] = field(
        default=64, metadata={"help": "The LoRA rank."}
    )
    lora_alpha: Optional[float] = field(
        default=32, metadata={"help": "The LoRA alpha."}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "The LoRA dropout."}
    )
    bits: Optional[int] = field(
        default=4, metadata={"help": "The number of bits to quantize to."}
    )
    double_quant: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use double quantization."}
    )
    quant_type: str = field(
        default="nf4", metadata={"help": "Quantization data type to use. [fp4, nf4]"}
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM."}
    )
    use_auth_token: str_or_bool = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token to download private/restricted models."}
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="Dahoas/full-hh-rlhf", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    block_size: Optional[int] = field(
        default=4096, metadata={"help": "The maximum length of the training sequence."}
    )
    multi_gpu: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use multiple GPUs."}
    )
    tensor_parallel: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use tensor parallelism. (Must be used with multi_gpu)"}
    )
    model_output_dir: Optional[str] = field(
        default="LLaMA/LoRA", metadata={"help": "The directory to save the model."}
    )

        
def get_device_map(model_name, id_=0, do_int8=False, do_int4=True):

    with init_empty_weights():
        config = LlamaConfig.from_pretrained(model_name)
        model =  AutoModelForCausalLM.from_config(config)

    d = {id_: "5000MiB"}
    d[1] = "4500MiB"
    d[2] = "4500MiB"
    d[3] = "4500MiB"
    d[4] = "4500MiB"
    d[5] = "4500MiB"
    d[6] = "4500MiB"
    d[7] = "6000MiB"
    dtype = torch.float16
    if do_int8:
        dtype = torch.int8
    elif do_int4:
        dtype = torch.int4
    device_map = infer_auto_device_map(
        model, max_memory=d, dtype=dtype, no_split_module_classes=["BloomBlock", "OPTDecoderLayer", "LLaMADecoderLayer", "LlamaDecoderLayer"]
    )
    print(device_map)
    del model
    return device_map


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if data_args.tensor_parallel == True and data_args.multi_gpu == False:
        raise ValueError("Tensor parallelism can only be used with multi_gpu.") 

    if data_args.multi_gpu == True:
        if data_args.tensor_parallel == True:
            # split the model across GPUs
            device_map = get_device_map(model_args.model_name_or_path)
        else:
            # stick a copy of the model on each GPU
            device_map = {"": accelerate.Accelerator().process_index}
    else:
        device_map = "auto"

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    model =  AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        load_in_4bit=model_args.bits == 4,
        load_in_8bit=model_args.bits == 8,
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=model_args.bits == 4,
            load_in_8bit=model_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.double_quant,
            bnb_4bit_quant_type=model_args.quant_type,
        ),
        torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)),
        trust_remote_code=model_args.trust_remote_code,
        use_auth_token=model_args.use_auth_token
    )

    model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, max_length=4096)
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    tokenizer.add_special_tokens({
        "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
        "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
        "unk_token": tokenizer.convert_ids_to_tokens(
            model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
        ),
    })


    # ### Prepare model for training
    #
    # Some pre-processing needs to be done before training such an int8 model using `peft`, therefore let's import an utiliy function `prepare_model_for_int8_training` that will:
    # - Cast the layer norm in `float32` for stability purposes
    # - Add a `forward_hook` to the input embedding layer to enable gradient computation of the input hidden states
    # - Enable gradient checkpointing for more memory-efficient training
    # - Cast the output logits in `float32` for smoother sampling during the sampling procedure
    
    # for param in model.parameters():
    #     param.requires_grad = False  # freeze the model - train adapters later
    #     if param.ndim == 1:
    #         # cast the small parameters (e.g. layernorm) to fp32 for stability
    #         param.data = param.data.to(torch.float16) #32) half precision seems to work just as well in practice
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # model.lm_head = CastOutputToFloat(model.lm_head)


    # model = prepare_model_for_int8_training(model) seemed to mess up training stability for some reason


    # ### Apply LoRA
    #
    # Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.

    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'] # edit with your desired target modules
    config = LoraConfig(
        r=model_args.r,
        lora_alpha=model_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    print_trainable_parameters(model_args, model)

    block_size = data_args.block_size


    ### Prepare dataset

    # Use this function to concatenate all texts from your dataset and generate chunks of block_size.
     # def group_texts(examples):
    #     # Concatenate all texts.
    #     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    #     total_length = len(concatenated_examples[list(examples.keys())[0]])
    #     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    #     # customize this part to your needs.
    #     if total_length >= block_size:
    #         total_length = (total_length // block_size) * block_size
    #     # Split by chunks of max_len.
    #     result = {
    #         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
    #         for k, t in concatenated_examples.items()
    #     }
    #     result["labels"] = result["input_ids"].copy()
    #     return result
    
    # def group_texts(examples):
    #     examples["labels"] = examples["input_ids"].copy()
    #     return examples

    def preprocess(sample):
        sample["chosen_sample"] = sample["prompt"] + sample["chosen"]
        return sample
    
    # tokenizer.add_bos_token = True
    # tokenizer.add_eos_token = False # Uncomment if you concatenate all texts from your dataset and generate chunks of block_size.
    # tokenizer.padding_side = "left"
    # tokenizer.truncation_side = "left"

    def tokenize(prompt):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=block_size,
            padding="max_length",
            add_special_tokens=True
        )
        return {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
        }


    ### Training
    dataset = load_dataset(data_args.dataset_name).map(preprocess)
    columns = dataset["train"].features
    # Use this for simple exmaple samples (conversation turns with dialogue history, instructions/responses, etc.)
    dataset = dataset.map(lambda samples: tokenize(samples["chosen_sample"]), batched=True, remove_columns=columns)
    # Use this to concatenate all texts from your dataset and generate chunks of block_size. (Books, etc.)
    #dataset = dataset.map(lambda samples: tokenizer(samples["chosen_sample"], padding=False, add_special_tokens=True), batched=True, remove_columns=columns)
    #dataset = dataset.map(group_texts, batched=True)

    # Train
    # model = torch.compile(model) # pytorch 2.0 but doesn't seem to work yet? (Should increase speed)
    if data_args.tensor_parallel == True:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    model.config.use_cache = True

    # Save model
    model.save_pretrained(data_args.model_output_dir)

if __name__ == "__main__":
    main()
