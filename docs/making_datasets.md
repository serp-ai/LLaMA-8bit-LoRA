# Making datasets

When creating a finetune dataset, you can either gather samples of inputs and outputs and construct the prompt template the model will expect during inference, or you can use a corpus of text to train the model to complete prompts in the style of the corpus. (Both are using the same training objective of predicting the next token in the sequence, but the way you construct the dataset and prompt the model is different.)

## Gathering samples

The simplest way to create a dataset is to gather a set of inputs and outputs using existing llms. A technique you can also use is to take an existing dataset (like Anthropic's HH for example) and use an llm to alter the outputs and/or inputs to create a new dataset in the tone/personality of you choosing. This is a good way to create a dialogue style dataset for a specific domain, like a specific company or a specific person. You can also generate datasets using few shot templates with a desired task/api usage and have an llm generate the outputs for you (inputs too if you'd like). (Examples of this are in the [examples](examples/) folder

## Training a model to complete prompts

The other way to create a dataset is to train a model to complete prompts. This is a good way to create a dataset for creative writing, idea generation, or any other task where you want the model to finish your prompt in the way that it was trained. To create these datasets you can just add start and end tokens to each body of text, concatenate them, chunk them by the desired sequence length, and then train a model.

### Details
- 30k - 50k samples is a good starting point for a dataset of this type. (10k or possibly less may also turn out well depending on the task and the quality of the data)
- If the model doesn't seem to be modeling your data well, you can try adding more data, training for more epochs, or revisting your dataset to see if there are actual patterns for the model to learn. (If you're using a corpus of text, you can try using a different corpus, or filtering out irrelevant text) (If they are instructions or dialogue, you can try making sure the inputs and outputs match your desired behavior)
