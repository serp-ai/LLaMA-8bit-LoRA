# Training
(If you would like to use flash attention and not afraid to go in to the transformers source code look below before starting the training)

## Requirements
If not using multi-gpu you should be able to download the requirements from pip but if you want to use multi-gpu you need to install the requirements from their github repos.
- transformers `pip install transformers` or install from https://github.com/huggingface/transformers `pip install git+https://github.com/huggingface/transformers.git` (if using flash attention you need to git clone the repo, edit the `modeling_llama.py` file, and then install from the local repo)
- peft `pip install peft` or install from https://github.com/huggingface/peft `pip install git+https://github.com/huggingface/peft.git`
- bitsandbytes `pip install bitsandbytes` if linux you're done, if windows you need to do the following steps:
    - download [libbitsandbytes_cuda116.dll](https://github.com/DeXtmL/bitsandbytes-win-prebuilt)
    - put libbitsandbytes_cuda116.dll in your bitsandbytes folder (usually in your python site-packages folder)
    - edit \bitsandbytes\cuda_setup\main.py:
        - search for:
            - `if not torch.cuda.is_available(): return 'libsbitsandbytes_cpu.so', None, None, None, None`
        - replace with:
            - `if torch.cuda.is_available(): return 'libbitsandbytes_cuda116.dll', None, None, None, None`
        - search for this twice:
            - `self.lib = ct.cdll.LoadLibrary(binary_path)`
        - replace with:
            - `self.lib = ct.cdll.LoadLibrary(str(binary_path))`

## Using custom datasets
You can either edit the training script directly to load your desired data and process it in to the correct format or if it is a dataset available in datasets you can use the `--dataset_name` and edit the training script to use the correct dataset keys. The section to edit for custom data loading are around line 210 in `finetune_peft_8bit.py`.

## Training
Recommended starting parameters:
- epochs: 2-4
- batch_size: Larger effective batch sizes like 128 or 256 are recommended (effective batch size = batch_size * gradient_accumulation_steps * num_gpus)
- block_size: 512 if not using flash attention, 600 if using flash attention (assuming 24gb of vram and batch of 2 with 64 gradient accumulation, can push it up to 2048 if you have more vram)
- learning_rate: 2e-4 (From LoRA paper)
- warmup_ratio: 0.06 (From LoRA paper)
- lr_scheduler_type: linear (From LoRA paper)
- weight_decay: 0.1 # worked well in experiments but may want to test as it's not based on any paper
- optim: adamw or adamw_torch_fused (need pytorch 2.0)

Launch the training script with your desired parameters.

Example launch command:
`python finetune_peft_8bit.py --num_train_epochs=2 --model_name_or_path=decapoda-research/llama-7b-hf --model_output_dir=LLaMA/LoRA/7B --output_dir=LLaMA/LoRA/train/7B --block_size=600 --per_device_train_batch_size=2 --gradient_accumulation_steps=64 --fp16=true --logging_steps=1 --log_level=info --learning_rate=2.0e-04 --lr_scheduler_type=linear --warmup_ratio=0.06 --weight_decay=0.1 --optim=adamw_torch_fused --evaluation_strategy=steps --save_strategy=steps --eval_steps=400 --save_steps=400 --output_dir="LoRA" --save_total_limit=3 --load_best_model_at_end=True --dataset_name=Dahoas/full-hh-rlhf --r=64 --lora_alpha=32 --lora_dropout=0.05`

Or if using multiple gpus (make sure you have the correct number of gpus in the `accelerate_config.yaml` file)
`accelerate launch --config_file=accelerate_config.yaml finetune_peft_8bit.py --multi_gpu=True --tensor_parallel=False --num_train_epochs=2 --model_name_or_path=decapoda-research/llama-7b-hf --model_output_dir=LLaMA/LoRA/7B --output_dir=LLaMA/LoRA/train/7B --block_size=600 --per_device_train_batch_size=2 --gradient_accumulation_steps=8 --fp16=true --logging_steps=1 --log_level=info --learning_rate=2.0e-04 --lr_scheduler_type=linear --warmup_ratio=0.06 --weight_decay=0.1 --optim=adamw_torch_fused --evaluation_strategy=steps --save_strategy=steps --eval_steps=400 --save_steps=400 --output_dir="LoRA" --save_total_limit=3 --load_best_model_at_end=True --remove_unused_columns=False --dataset_name=Dahoas/full-hh-rlhf --r=64, --lora_alpha=32, --lora_dropout=0.05`

Or if using multiple gpus and tensor parrallelism (go in to the main training file and edit the `get_device_map` function with your number of devices and desired memory allocation)
`python finetune_peft_8bit.py --num_train_epochs=2 --multi_gpu=True --tensor_parallel=True --model_name_or_path=decapoda-research/llama-65b-hf --model_output_dir=LLaMA/LoRA/65B --output_dir=LLaMA/LoRA/train/65B --block_size=600 --per_device_train_batch_size=2 --gradient_accumulation_steps=64 --fp16=true --logging_steps=1 --log_level=info --learning_rate=2.0e-04 --lr_scheduler_type=linear --warmup_ratio=0.06 --weight_decay=0.1 --optim=adamw_torch_fused --evaluation_strategy=steps --save_strategy=steps --eval_steps=400 --save_steps=400 --output_dir="LoRA" --save_total_limit=3 --load_best_model_at_end=True --dataset_name=Dahoas/full-hh-rlhf --r=64 --lora_alpha=32 --lora_dropout=0.05`


## Flash attention (optional - pytorch 2.0 required)
If you want to use flash attention, you need to change the source code of the transformers library. You can find the source code [here](https://github.com/huggingface/transformers). You need to change the `modeling_llama.py` file. located at `transformers/src/transformers/models/llama/modeling_llama.py`. You need to change the `LlamaAttention` class to the following:

```python
class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {num_heads})."
            )
        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            num_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            num_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(hidden_size, hidden_size))
                                  .view(1, 1, hidden_size, hidden_size))

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].shape[-2]
            kv_seq_len += offset
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, offset=offset)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if self.flash:
            attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask)
        else:
            past_key_value = (key_states, value_states)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
```

The changes are
```python
self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
if not self.flash:
    print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
    self.register_buffer("bias", torch.tril(torch.ones(hidden_size, hidden_size))
                            .view(1, 1, hidden_size, hidden_size))
```
and
```python
if self.flash:
    attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask)
else:
    ...
```

After those changes, pip install the local repo by cd'ing in to the transformers repo (root directory) and then running `pip install .`. Then, add/uncomment the following to the imports of the training script:
```python
import torch.backends.cuda
torch.backends.cuda.enable_flash_sdp(enabled=True)
```