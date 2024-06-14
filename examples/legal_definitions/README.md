# Legal Definitions

The use case in this directory extracts terms and their definitions from legal documents. It is based on an in-house legal analytics project.

By default it will test different chunk sizes. To execute only one experiment please comment out other chunk sizes in `main.py`.

## Data

The dataset can be found in the file `dataset/legal_definitions.json`. It consists of text chunks as well as expected terms to be found (the "ground truth").

## Prompt Template

The prompt template can be found in the file `prompt_scheme.txt`.

## Runtime / Cost Estimation

The samples have been generated with a temperature of 0.25. The temperature can be adjusted in your `config.json`.
We estimate a compute time of 90 minutes with an NVIDIA A100-SXM-40GB for each experiment.

Based on the experiment the total estimated costs are $7, $11 and $18:
- GPT4-o: $2.25, $3.5, $5.75
- GPT4-turbo: $4.5, $7, $10.5
- GPT3.5: $0.15, $0.5, $1.5
