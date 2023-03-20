# Weight merging
You can merge the lora with the foundation for further training such as the ppo stage in rlhf, to have faster inference, to quantize the model to 4/3/2 bit, or to just not have to deal with juggling the two models.

All you have to do is run this command
- `python merge_adapter_weights --model_name=<your_adapter_local_path_or_huggingface_model_id_goes_here> --output_dir=<where_to_save_the_merged_model>`

The config of the LoRA saves the path to the foundation model, so you can just run the command without any extra arguments and it will merge the weights for you.