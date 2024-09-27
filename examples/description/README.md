# Distinguishing Similar and Different Text Passages

The use case in this directory analyzes, whether a verification method is able to clearly distinguish two passages of text that either look
similar, but come with very different meanings ("different") or look different, but have similar or identical meanings ("similar").

## Data

The list of topics for the different subtask can be found in `different_topics_list` list in the `different/main.py` file.
There are two lists of topics for the similar subtask: `precise` and `generic`. Both lists (`precise_topics` and `general_topics`) can be found
in the `similar/main.py` file.

## Prompt Templates

The prompt templates for the subtasks can be found in `different/prompt_scheme.txt` and `similar/prompt_scheme.txt` respectively.

## Runtime / Cost Estimation

The samples have been generated with a temperature of 1.0. The temperature can be adjusted in your `config.json`.
We estimate a compute time of 90 minutes with an NVIDIA Tesla V100-PCIE-32GB for each subtask.
   
The total estimated costs are $1.55 for each subtask:
- GPT4-o: $0.5
- GPT4-turbo: $1
- GPT3.5: $0.05
