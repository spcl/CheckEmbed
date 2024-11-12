# WikiBio Benchmark

In addition to our own datasets, we use one more benchmark: WikiBio. Specificly, we use a subset of the WikiBio dataset (Lebret et al., 2016) that was modified by Manakul et al. (2023) for their evaluation of SelfCheckGPT. It consists of 238 documents based on Wikipedia articles, that were used to generate samples in which hallucinations were introduced. Each sentence of those samples were manually labeled as either “major inaccurate”, “minor inaccurate”, or “accurate”.

## Data

The dataset and the conversion of sentence score into passage score can be found in the `data` folder.
To freshly download the dataset and recompute the passage scores, run the following commands:

```bash
cd data
python3 download.py
python3 passage_scores.py
```

## Runtime / Cost Estimation

The estimated compute time for running the evaluation is approximately 36 hours on an NVIDIA A100-SXM-40GB.

The sample step is skipped, since the samples are already provided in the dataset. The cost only occurs for the embedding with the OpenAI models.

The embedding model from OpenAI has a cost of $0.13 / 1M tokens. Resulting in an approximate cost of $0.65 for the evaluation.
