# Hallucination

The use case in this directory detects small fine-grained hallucinations, such
as mistakes in individual facts. The use case is based on the description of different scientific topics.
For each topic considered, the ground truth is generated using a
special prompt `prompt_scheme_ground_truth.txt`, which gathers 10 samples from the LLM by asking for a correct description of that specific topic.
The LLM is also tasked to provide errors for that topic, in the range from 1 to 10.
These errors are then incorporated separately into the description, so that the number of errors inside the description varies between 1 to 10.
These error-ridden descriptions are then sampled with an LLM and compared against the zero error original description via the CheckEmbed pipeline.

## Data

The list of topics can be found in `topics_list` list in the `main.py` file.

## Prompt Templates

The prompt templates can be found in the files `prompt_scheme.txt` and `prompt_scheme_ground_truth.txt`.

## Runtime / Cost Estimation

The samples have been generated with a temperature of 1.0. The temperature can be adjusted in your `config.json`.
We estimate a compute time of 20 hours with an NVIDIA GH200.

The total estimated costs are $35:
- GPT4-o: $33
- GPT3.5: $2
