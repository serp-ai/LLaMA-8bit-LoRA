# Code used from https://github.com/CarperAI/trlx/blob/main/examples/hh/sft_hh.py https://github.com/lvwerra/trl/blob/main/examples/sentiment/scripts/gpt-neox-20b_peft/gpt-neo-20b_sentiment_peft.py

from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import os
import torch
import torch.nn as nn
import transformers
import accelerate
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from accelerate import init_empty_weights, infer_auto_device_map
from transformers import AutoConfig, set_seed

# Uncomment this line to enable flash attention (model source code must be modified)
# import torch.backends.cuda
# torch.backends.cuda.enable_flash_sdp(enabled=True)


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
        default="decapoda-research/llama-7b-hf",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
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



@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="Dahoas/full-hh-rlhf", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    block_size: Optional[int] = field(
        default=2048, metadata={"help": "The name of the dataset to use (via the datasets library)."}
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


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
        
def get_device_map(model_name, id_=0, do_int8=True):

    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)

    d = {id_: "5000MiB"}
    d[1] = "4500MiB"
    d[2] = "4500MiB"
    d[3] = "4500MiB"
    d[4] = "4500MiB"
    d[5] = "4500MiB"
    d[6] = "4500MiB"
    d[7] = "6000MiB"
    device_map = infer_auto_device_map(
        model, max_memory=d, dtype=torch.int8 if do_int8 else torch.float16, no_split_module_classes=["BloomBlock", "OPTDecoderLayer", "LLaMADecoderLayer"]
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
            get_device_map(model_args.model_name_or_path)
        else:
            # stick a copy of the model on each GPU
            device_map = {"": accelerate.Accelerator().process_index}
    else:
        device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=True,
        device_map=device_map
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, max_length=2048)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token


    # ### Prepare model for training
    #
    # Some pre-processing needs to be done before training such an int8 model using `peft`, therefore let's import an utiliy function `prepare_model_for_int8_training` that will:
    # - Cast the layer norm in `float32` for stability purposes
    # - Add a `forward_hook` to the input embedding layer to enable gradient computation of the input hidden states
    # - Enable gradient checkpointing for more memory-efficient training
    # - Cast the output logits in `float32` for smoother sampling during the sampling procedure
    
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float16) #32) half precision seems to work just as well in practice

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float16) #32) half precision seems to work just as well in practice
    model.lm_head = CastOutputToFloat(model.lm_head)


    # model = prepare_model_for_int8_training(model) seemed to mess up training stability for some reason


    # ### Apply LoRA
    #
    # Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.

    target_modules = None
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'] # edit with your desired target modules
    config = LoraConfig(
        r=data_args.r, lora_alpha=data_args.lora_alpha, target_modules=target_modules, lora_dropout=data_args.lora_dropout, bias="none", task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    block_size = data_args.block_size


    ### Prepare dataset

    # Use this function to concatenate all texts from our dataset and generate chunks of block_size.
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
    
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = False
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

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
    # Use this to concatenate all texts from our dataset and generate chunks of block_size. (Books, etc.)
    #dataset = dataset.map(lambda samples: tokenizer(samples["chosen_sample"], padding=False, add_special_tokens=True), batched=True, remove_columns=columns)
    #dataset = dataset.map(group_texts, batched=True)


    model.gradient_checkpointing_enable()
    tokenizer.padding = True
    # model = torch.compile(model) # pytorch 2.0 but doesn't seem to work yet? (Should increase speed)
    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    model.save_pretrained(data_args.model_output_dir)

if __name__ == "__main__":
    main()
