# Legal Definitions

The use case in this directory extracts terms and their definitions from legal documents. It is based on an in-house legal analytics project.

We use this example also for an ablation study by varying the chunk sizes that are processed in a single step.
An increase in chunk size means that more terms and their definitions need to be extracted at a time.
The general assumption is that the LLM will perform worse if the processed document size increases, which should be reflected in the resulting CheckEmbed scores.
If you wish the run the original use case with a single chunk size, please comment out the lines 233 to 241 in `main.py`.

## Data

The dataset can be found in the file `dataset/legal_definitions.json`. It consists of text chunks as well as expected terms to be found (the "ground truth").

## Prompt Template

The prompt template can be found in the file `prompt_scheme.txt`.

## Runtime / Cost Estimation

The samples have been generated with a temperature of 0.25. The temperature can be adjusted in your `config.json`.
We estimate a compute time of 90 minutes with an NVIDIA A100-SXM-40GB for each experiment.

Based on the experiment the total estimated costs are $7 (1 chunk), $11 (2 chunks) and $18 (4 chunks):
- GPT4-o: $2.25, $3.5, $5.75
- GPT4-turbo: $4.5, $7, $10.5
- GPT3.5: $0.15, $0.5, $1.5
