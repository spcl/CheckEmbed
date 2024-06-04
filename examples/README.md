# Examples

This directory contains scripts for running various examples using the CheckEmbed package. Each script is a standalone Python program that sets up and runs a particular example.

Please refer to the individual example directories for more information on the specific example, specifically the `main.py` file, which is almost ready to be executed.


## General Information

In each `main.py` file, the following parameters need to be set up for the desired environment:
- Check that the `config_path` variable is set up correctly.
- Choose the language model(s) to evaluate.
- Choose the embedding model(s).
- Check the `device` and `batch_size` parameters for the embeddings models and scheduler.
- Modify the `startingPoint` parameter of `scheduler.run(...)` to influence which stages will be executed:
  - `StartingPoint.PROMPT`: prompt generation, sample generation, embedding generation and evaluation (plotting)
  - `StartingPoint.SAMPLES`: sample generation, embedding generation and evaluation (plotting)
  - `StartingPoint.EMBEDDINGS`: embedding generation and evaluation (plotting)
- If you want to use the `Alibaba-NLP/gte-Qwen1.5-7B-instruct` embedding model, please add your Huggingface access token to respective initialisation call.

Once everything is set up, change into the desired example folder and execute:
```
python3 main.py
```


## Scheduler Setup

The file [scheduler.py](/CheckEmbed/scheduler/scheduler.py) contains specific documentation for each parameter.

```python
scheduler = Scheduler(
    current_dir,
    logging_level = logging.DEBUG,

    # Adjust the budget based on the estimations documented for each example.
    # If the budget is too low, the execution of the pipeline will be stopped as soon as the limit is detected.
    budget = 12,
    parser = customParser,

    # Update to include more or fewer LLMs / embedding models.
    lm = [gpt4_o, gpt4, gpt3],
    embedding_lm = [embedd_large, sfrEmbeddingMistral, e5mistral7b, gteQwen157bInstruct],

    # Operations to be executed during the evaluation stage.
    operations = [bertPlot, selfCheckGPTPlot, rawEmbeddingHeatPlot, checkEmbedPlot],
)

# The order of lm_names and embedding_lm_names should be the same
# as the order of the language models and embedding language models respectively.
scheduler.run(
    # If an error occurs, the starting point can be adjusted to avoid recomputation.
    startingPoint = StartingPoint.PROMPT,

    # Indicate which operations to run.
    bertScore = True,
    selfCheckGPT = True,
    checkEmbed: bool = True,
    ground_truth = False,
    spacy_separator = True,

    # Number of samples per prompt example.
    num_samples = 10,

    # Change accordingly to changes of the lm / embedding_lm parameters of the Scheduler.
    lm_names = ["gpt4-o", "gpt4-turbo", "gpt"],
    embedding_lm_names = ["gpt-embedding-large", "sfr-embedding-mistral", "e5-mistral-7b-instruct", "gte-Qwen15-7B-instruct"],

    # Do not modify
    bertScore_model = "microsoft/deberta-xlarge-mnli",

    # It may be necessary to reduce the batch size if the model is too large, with 8GB of GPU VRAM we suggest the use of batch_size = 1.
    batch_size = 64,
    device = "cuda" # or "cpu" "mps" ...
)
```
