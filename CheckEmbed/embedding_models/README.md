# Embedding Models

The Embedding Models module is responsible for managing the embedding models.

Currently, the framework supports the following embedding models:

- text-embedding-large / small (remote - OpenAI API)
- Salesforce/SFR-Embedding-Mistral (local - GPU with 32GB VRAM recommended, model size is roughly 26GB)
- intfloat/e5-mistral-7b-instruct (local - GPU with 32GB VRAM recommended, model size is roughly 26GB)
- Alibaba-NLP/gte-Qwen1.5-7B-instruct (local - GPU with 32GB VRAM recommended, model size is roughly 26GB)
- NovaSearch/stella_en_1.5B_v5 (local - GPU with 12GB VRAM recommended, model size is roughly 6GB)
- NovaSearch/stella_en_400M_v5 (local - GPU with 4GB VRAM recommended, model size is roughly 2GB)
- openai/clip-vit-large-patch14 (local - GPU with 4GB VRAM recommended, model size is roughly 2GB)

The following sections describe how to instantiate individual models and how to add new models to the framework.

## Embedding Model Instantiation

- Create a copy of `config_template.json` named `config.json` in the CheckEmbed folder. (Not necessary for local models)
- Fill in the configuration details based on the used model (below).

### Embedding-Text-Large / Embedding-Text-Small

- Adjust the predefined `gpt-embedding-large` or `gpt-embedding-small` configurations or create a new configuration with an unique key.

| Key                 | Value                                                                                                                                                                                                                                                                                                                                                               |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model_id            | Model name based on [OpenAI model overview](https://platform.openai.com/docs/models/overview).                                                                                                                                                                                                                                                                      |
| name                | Name used for CheckEmbed output files. We suggest to use the default names for local models.                                                                                                                                                                                                                                                                        |
| token_cost          | Price per 1000 tokens based on [OpenAI pricing](https://openai.com/pricing), used for calculating cumulative price per LLM instance.                                                                                                                                                                                  |
| encoding            | String indicating the format to return the embeddings in. Can be either float or base64. More information can be found in the [OpenAI API reference](https://platform.openai.com/docs/api-reference/embeddings/create#embeddings-create-encoding_format). |
| dimension           | Number indicating output dimension for the embedding model. More information can be found in the [OpenAI model overview](https://platform.openai.com/docs/models/overview).                                                                                                       |
| organization        | Organization to use for the API requests (may be empty).                                                                                                                                                                                                                                                                                                            |
| api_key             | Personal API key that will be used to access the OpenAI API.                                                                                                                                                                                                                                                                                                        |

- Instantiate the embedding model based on the selected configuration key (predefined / custom).
  - `max_concurrent_request` is by default 10. Adjust the value based on your tier [rate limits](https://platform.openai.com/docs/guides/rate-limits).

```python
embedding_lm = language_models.EmbeddingGPT(
                    config_path,
                    model_name = <configuration-key>,
                    cache = <False | True>,
                    max_concurrent_requests = <int number>
                )
```

### Local Models

The framework currently supports the following local models: `Salesforce/SFR-Embedding-Mistral`, `intfloat/e5-mistral-7b-instruct`, `Alibaba-NLP/gte-Qwen1.5-7B-instruct`, `NovaSearch/stella_en_1.5B_v5`, `NovaSearch/stella_en_400M_v5` and `openai/clip-vit-large-patch14`.

- Instantiate the embedding model based on the owned device.
- Device can be specified in the `Scheduler`, more [here](/CheckEmbed/scheduler/scheduler.py)

```python
sfrEmbeddingMistral = language_models.SFREmbeddingMistral(
                          model_name = "Salesforce/SFR-Embedding-Mistral",
                          cache = False,
                          batch_size = 64,
                      )

e5mistral7b = language_models.E5Mistral7b(
                    model_name = "intfloat/e5-mistral-7b-instruct",
                    cache = False,
                    batch_size = 64,
                )

gteQwen157bInstruct = language_models.GteQwenInstruct(
                            model_name = "Alibaba-NLP/gte-Qwen1.5-7B-instruct",
                            cache = False,
                            access_token = "", # Add your access token here (Hugging Face)
                            batch_size = 1, # Unless you have more than 32GB of GPU VRAM at your disposal use 1.
                        )

stella_en_15B_v5 = embedding_models.Stella(
        model_name = "NovaSearch/stella_en_1.5B_v5",
        cache = False,
        batch_size = 64,
    )

stella_en_400M_v5 = embedding_models.Stella(
        model_name = "NovaSearch/stella_en_400M_v5",
        cache = False,
        batch_size = 64,
    )

clip_vit_large = embedding_models.ClipVitLarge(
        model_name = "openai/clip-vit-large-patch14",
        cache = False,
    )
```

## Adding Embedding Models

More embedding models can be added by following these steps:

- Create new class as a subclass of `AbstractEmbeddingModel`.
- Use the constructor for loading the configuration and instantiating the embedding model (if needed).

```python
class CustomLanguageModel(AbstractEmbeddingModel):
    def __init__(
        self,
        config_path: str = "",
        model_name: str = "text-embedding-large",
        name: str = "CustomLanguageModel",
        cache: bool = False
    ) -> None:
        super().__init__(config_path, model_name, name, cache)
        self.config: Dict = self.config[model_name]
        
        # Load data from configuration into variables if needed

        # Instantiate model if needed
```

- Implement the `load_model`, `unload_model` and `generate_embedding` abstract methods that are used to load/unload the model from the GPU (if necessary) and get a list of embeddings from the model (remote API call or local model inference) respectively.

```python
def load_model(self, device: str = None) -> None:
    """
    Load the model and tokenizer based on the given model name.

    :param device: The device to load the model on. Defaults to None.
    :type device: str
    """

def unload_model(self) -> None:
    """
    Unload the model and tokenizer.
    """

def generate_embedding(
        self,
        input: Union[List[Any], Any]
    ) -> List[float]:
    # Call model and retrieve an embedding
    # Return model response
```
