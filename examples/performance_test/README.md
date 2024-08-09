# Performance Testing

This directory contains scripts and configurations to evaluate the performance of CheckEmbed on various embedding models in comparison to SelfCheckGPT and BERTScore.
The script generates text samples and analyzes the performance while varying the text size, i.e. the number of tokens in the text, and the number of samples.
Varying the number of tokens to embed gives insights on the overall efficiency of the different embedding models used by CheckEmbed, SelfCheckGPT and BERTScore, while varying the sample number examines the the scalability of the respective pipelines.

By default, the script tests multiple text sizes, ranging from 200 to 4000 tokens in steps of 200, as well as different number of samples (2, 4, 6, 8 and 10).

## Data

The dataset used to generate text samples is created using the `Faker` library. Samples of varying lengths are generated and saved in a JSON format in directories (`2_samples`, `4_samples`, etc.) corresponding to the number of samples..

Once desired evaluation is finished, `data_extractor.py` can be used (and/or modified) to parse the runtime logs and create a single JSON file, which contains all runtime measurements.
```python
python3 data_extractor.py
```

## Runtime / Cost Estimation

The estimated compute time for running the evaluation with a specific number of samples is approximately 5-8 hours on an NVIDIA A100-SXM-40GB.

Experiments skip the sample phase resulting in lower cost deriving from the API calls to OpenAI text-embedding-large.
The embedding model from OpenAI has a cost of $0.13 / 1M tokens.

### Example
In the following, we calculate the total cost for running the runtime measurements with the default parameters:
- varying the number of samples from 2 to 10 in increments of 2
- varying the text size from 200 to 4000 tokens in steps of 200 tokens
- 20 prompts, meaning 20 datapoints for each specific combination of number of samples and number of tokens

The total costs are $3.28:
- total number of samples per text size: (2 + 4 + 6 + 8 + 10) * 20 = 30 * 20 = 600
- total number of tokens:
  - 200 tokens: 200 * 600 = 120K
  - 400 tokens: 400 * 600 = 240K
  - ...
  ---
  - 25.2M tokens
- 25.2M tokens * $0.13 / 1M tokens = $3.28
