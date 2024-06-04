# Legal Definitions

The use case in this directory extracts terms and their definitions from legal documents. It is based on an in-house legal analytics project.

## Data

The dataset can be found in the file `dataset/legal_definitions.json`. It consists of text chunks as well as expected terms to be found (the "ground truth").

## Prompt Template

The prompt template can be found in the file `prompt_scheme.txt`.

## Runtime / Cost Estimation

The samples have been generated with a temperature of 0.25. The temperature can be adjusted in your `config.json`.
We estimate a compute time of 45 minutes with an NVIDIA Tesla V100-PCIE-32GB.

The total estimated costs are $4:
- GPT4-o: $1.25
- GPT4-turbo: $2.5
- GPT3.5: $0.15
