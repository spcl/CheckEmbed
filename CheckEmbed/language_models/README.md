# Language Models

The Language Models module is responsible for managing the large language models (LLMs).

Currently, the framework supports the following LLMs models:
- GPT-4 / GPT-3.5 (remote - OpenAI API)

The following sections describe how to instantiate individual models and how to add new models to the framework.

## LLM Instantiation
- Create a copy of `config_template.json` named `config.json`. (Not necessary for local models)
- Fill configuration details based on the used model (below).

### GPT-4 / GPT-3.5
- Adjust the predefined `gpt-3.5-turbo-0125`, `gpt-4`, `gpt-4-turbo` or `gpt-4o` configurations or create a new configuration with an unique key.

| Key                 | Value                                                                                                                                                                                                                                                                                                                                                               |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model_id            | Model name based on [OpenAI model overview](https://platform.openai.com/docs/models/overview).                                                                                                                                                                                                                                                                      |
| prompt_token_cost   | Price per 1000 prompt tokens based on [OpenAI pricing](https://openai.com/pricing), used for calculating cumulative price per LLM instance.                                                                                                                                                                                                                         |
| response_token_cost | Price per 1000 response tokens based on [OpenAI pricing](https://openai.com/pricing), used for calculating cumulative price per LLM instance.                                                                                                                                                                                                                       |
| temperature         | Parameter of OpenAI models that controls the randomness and the creativity of the responses (higher temperature = more diverse and unexpected responses). Value between 0.0 and 2.0, default is 1.0. More information can be found in the [OpenAI API reference](https://platform.openai.com/docs/api-reference/completions/create#completions/create-temperature).     |
| max_tokens          | The maximum number of tokens to generate in the chat completion. Value depends on the maximum context size of the model specified in the [OpenAI model overview](https://platform.openai.com/docs/models/overview). More information can be found in the [OpenAI API reference](https://platform.openai.com/docs/api-reference/chat/create#chat/create-max_tokens). |
| stop                | String or array of strings specifying sequences of characters which if detected, stops further generation of tokens. More information can be found in the [OpenAI API reference](https://platform.openai.com/docs/api-reference/chat/create#chat/create-stop).                                                                                                       |
| organization        | Organization to use for the API requests (may be empty).                                                                                                                                                                                                                                                                                                            |
| api_key             | Personal API key that will be used to access OpenAI API.                                                                                                                                                                                                                                                                                                            |

- Instantiate the language model based on the selected configuration key (predefined / custom).
    - `max_concurrent_request` is by default 10. Adjust the value based on your tier [rate limits](https://platform.openai.com/docs/guides/rate-limits).
```python
lm = language_models.ChatGPT(
            config_path,
            model_name = <configuration-key>,
            cache = <False | True>,
            max_concurrent_requests = <int number>
        )
```


## Adding LLMs
More LLMs can be added by following these steps:
- Create a new class as a subclass of `AbstractLanguageModel`.
- Use the constructor for loading the configuration and instantiating the language model (if needed).
```python
class CustomLanguageModel(AbstractLanguageModel):
    def __init__(
        self,
        config_path: str = "",
        model_name: str = "llama7b-hf",
        cache: bool = False
    ) -> None:
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        
        # Load data from configuration into variables if needed

        # Instantiate LLM if needed
```
- Implement `query` abstract method that is used to get a list of responses from the LLM (call to remote API or local model inference).
```python
def query(
        self,
        query: str,
        num_query: int = 1
    ) -> Any:
    # Support caching 
    # Call LLM and retrieve list of responses - based on num_query    
    # Return LLM response structure (not only raw strings)    
```
- Implement `get_response_texts` abstract method that is used to get a list of raw texts from the LLM response structure produced by `query`.
```python
def get_response_texts(
        self, 
        query_response: Union[List[Any], Any]
    ) -> List[str]:
    # Retrieve list of raw strings from the LLM response structure 
```
