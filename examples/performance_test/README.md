# Performance Testing

This directory contains scripts and configurations for evaluating the performance of CheckEmbed on various embedding models vs SelfCheckGPT vs BERTScore. The primary objective is to generate text samples and analyze performance when varying text size (number of token) and sample number.
The first will give insights on the overall efficiency of the diffrent embedding models used by CheckEmbed, SelfCheckGPT and BERTScore. The second one will show the scalability of this pipelines.

By default, the script tests multiple text sizes, going from a token size of 200 to 4000 in steps of 200. It is also testing different number of samples from 2 to 10 in steps of 2.

## Data

The dataset used for generating text samples is created using the `Faker` library. Samples of varying lengths are generated and saved in JSON format in corresponding directories (`2_samples`, `4_samples`, etc.).

When the desired tests are all finished you can modify and use the `data_extractor.py` to parse all runtimes log and create a single json file.
```python
python3 data_extractor.py
```

## Runtime / Cost Estimation

The samples have been generated with varying lengths and batch sizes. The estimated compute time for running the evaluations is approximately 5-8 hours on an NVIDIA A100-SXM-40GB for each different number of sample test.

Experiments skip the sample phase resulting in lower cost deriving from the API calls to OpenAI text-embedding-large.
The embedding model from OpenAI has a cost of $0.13 / 1M tokens.

### Example
Considering the default behaviour of testing from 2 to 10 samples with everyone using from 200 to 4000 tokens and 20 prompts the total cost is going to be:
- Samples for every token dimension: $(2 + 4 + 6 + 8 + 10) * 20 = 30 * 20 = 600$
- Total Number of Tokens:
    - 200 tokens: $200 * 600 = 120k$
    - 400 tokens: $400 * 600 = 240k$
    - ...
    ---
    - 25kk tokens

Total cost: $4.00
