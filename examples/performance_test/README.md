# Performance Testing

This directory contains scripts and configurations to evaluate the performance, specifically the runtime, of CheckEmbed on various embedding models in comparison to SelfCheckGPT and BERTScore.
The script generates input text for each datapoint while varying the sizes of these texts, i.e. the number of tokens in the text, as well as the number of samples for each datapoint and measures the runtime performance of the embedding and the operations.
The samples of a datapoint are all generated locally via script instead of querying an LLM.
Varying the number of tokens to embed gives insights on the overall efficiency of the different embedding models used by CheckEmbed, SelfCheckGPT and BERTScore, while varying the sample number examines the the scalability of the respective pipelines.

By default, the script tests multiple text sizes, ranging from 200 to 4000 tokens in steps of 200, as well as different number of samples (2, 4, 6, 8 and 10).

## Data

The dataset with the generated text samples is created using the `Faker` library. Samples of varying lengths are generated and stored in a JSON format in directories (`2_samples`, `4_samples`, etc.) corresponding to the number of samples.

Once the evaluation is finished, `data_extractor.py` can be used (and/or modified) to aggregate the runtime logs and write the results into a single JSON file containing all runtime measurements.
```python
python3 data_extractor.py
```

The extracted JSON file has the following structure in general:
```json
{
  "#_samples": {   //2_samples, 4_samples...
    "embedding": {
      "embedding_model_name": {   //gpt-embedding-large, sfr-embedding-mistral...
        "#tokens": "time",
        "#tokens": "time",
        //...
      },
      //more embeddings...
    },
    "bertscore": {
      "#tokens": "time",
      //...
    },
    "selfcheckgpt": {
      "#tokens": "time",
      //...
    },
    "checkembed": {
      "embedding_model_name": {   //gpt-embedding-large, sfr-embedding-mistral...
        "#tokens": "time",
        "#tokens": "time",
        //...
      },
      //more embeddings...
    },
    "operations": {} //To customize.
  },
  //additional number of samples...
} 
```
The runtime is reported in seconds.

The extracted data can be visualized with the help of the provided plotting script:
```python
python3 plot.py
```

## Runtime / Cost Estimation

The estimated compute time for running the evaluation is approximately 24 hours on an NVIDIA A100-SXM-40GB.

The sample step is only emulated for these runtime measurements to avoid the cost of calling the LLM for the sampling, so cost only occur for the embedding with the OpenAI models.

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
