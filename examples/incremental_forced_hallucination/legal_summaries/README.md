# Hallucination

The use case in this directory detects small fine-grained hallucinations, such
as mistakes in individual facts. The use case is based on the summarization of different legal text chunks.
For each chunk considered, the ground truth is generated using a
special prompt `prompt_scheme_ground_truth.txt`, which gathers 10 samples from the LLM by asking for a correct summarization of that chunk.
The LLM is also tasked to provide errors for that chunk, in the range from 1 to 10.
These errors are then incorporated separately into the summary, so that the number of errors inside the summary varies between 1 to 10.
These error-ridden summary are then sampled with an LLM and compared against the zero error original summary via the CheckEmbed pipeline.

## Data

The dataset can be found in the file `dataset/legal_definitions.json`. It consists of text chunks to be summarized.

## Prompt Templates

The prompt templates can be found in the files `prompt_scheme.txt` and `prompt_scheme_ground_truth.txt`.

## Runtime / Cost Estimation

The samples have been generated with a temperature of 0.25. The temperature can be adjusted in your `config.json`.
We estimate a compute time of 20 hours with an NVIDIA GH200.

The total estimated costs are $35:
- GPT4-o: $33
- GPT3.5: $2
