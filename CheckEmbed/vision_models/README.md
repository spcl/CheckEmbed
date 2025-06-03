# Vision Models

The Vision Models module is responsible for managing the vision models.

Currently, the framework supports the following vision model:

- stabilityai/stable-diffusion-3.5-medium (local - GPU with 12GB VRAM recommended, model size is roughly 6GB )

The following sections describe how to instantiate the model and how to add new models to the framework.

## Vision Model Instantiation

If your model needs a configuration file, follow these steps:

- Create a copy of `config_template.json` named `config.json` in the CheckEmbed folder. (Not necessary for local models)
- Fill configuration details based on the used model.

### Local Models

The framework currently supports the following local model: `stabilityai/stable-diffusion-3.5-medium`.

- Instantiate the vision model based on the owned device.
- Device can be specified in the `Scheduler`, more [here](/CheckEmbed/scheduler/scheduler.py).

```python
stable_diffusion = vision_models.StableDiffusion3(
        model_name = "stabilityai/stable-diffusion-3.5-medium",
        cache = False,
        data_type = torch.bfloat16,
        num_inference_steps = 40,
        guidance_scale = 4.5
    )
```

## Adding Vision Models

More vision models can be added by following these steps:

- Create new class as a subclass of `AbstractVisionModel`.
- Use the constructor for loading the configuration and instantiating the vision model (if needed).

```python
class CustomVisionModel(AbstractVisionModel):
    def __init__(
        self,
        config_path: str = "",
        model_name: str = "official model-name",
        name: str = "CustomVisionModel",
        cache: bool = False
    ) -> None:
        super().__init__(config_path, model_name, name, cache)
        self.config: Dict = self.config[model_name]

        # Load data from configuration into variables if needed

        # Instantiate model if needed
```

- Implement the `load_model`, `unload_model` and `generate_image` abstract methods that are used to load/unload the model from the GPU (if necessary) and get a list of images from the model (remote API call or local model inference) respectively.

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

def generate_image(
    self,
    input: Union[List[str], str]
    ) -> List[Image]:
    # Call model and retrieve an Image
    # Return model response
```
