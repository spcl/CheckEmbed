# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main authors: Lorenzo Paleari
#               Eric Schreiber

import gc
from typing import List, Union

import torch
from diffusers import StableDiffusion3Pipeline
from PIL.Image import Image
from tqdm import tqdm

from CheckEmbed.vision_models import AbstractVisionModel


class StableDiffusion3(AbstractVisionModel):
    """
    The StableDiffusion3 class handles interactions with the Stable Diffusion 3.5 Medium model using the provided configuration.

    Inherits from the AbstractVisionModel class and implements its abstract methods.
    """

    def __init__(
        self, model_name: str = "", name: str = "stable-diffusion3.5-medium", cache: bool = False, data_type: torch.dtype = torch.bfloat16, num_inference_steps: int = 40, guidance_scale: float = 4.5
    ) -> None:
        """
        Initialize the StableDiffusion3 instance with configuration, model details, and caching options.

        :param model_name: Name of the model, which is used to select the correct configuration. Defaults to "".
        :type model_name: str
        :param name: Name used for output files. Defaults to "stable-diffusion3.5-medium".
        :type name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        :param data_type: The data type for the model, typically torch.bfloat16 or torch.float32. Defaults to torch.bfloat16.
        :type data_type: torch.dtype
        :param num_inference_steps: The number of inference steps for image generation. Defaults to 40.
        :type num_inference_steps: int
        :param guidance_scale: The guidance scale for image generation, which controls the adherence to the prompt. Defaults to 4.5.
        :type guidance_scale: float
        """
        super().__init__(model_name=model_name, name=name, cache=cache)
        self.data_type = data_type
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

    def load_model(self, device: str = None) -> None:
        """
        Load the model and tokenizer based on the given model name.

        :param device: The device to load the model on. Defaults to None.
        :type device: str
        """

        self.model = StableDiffusion3Pipeline.from_pretrained(self.model_name, torch_dtype=self.data_type)
        self.model = self.model.to(device)

    def unload_model(self) -> None:
        """
        Unload the model and tokenizer.
        """
        del self.model

        gc.collect()
        torch.cuda.empty_cache()

        self.model = None

    def generate_image(self, input: Union[List[str], str]) -> List[Image]:
        """
        Generate images based on the input prompts using the Stable Diffusion 3.5 Medium model.

        This method takes a list of prompts or a single prompt string, generates images for each prompt,
        and returns a list of generated images. The prompts are processed in batches to optimize performance.

        :param input: A list of prompts or a single prompt string to generate images for.
        :type input: Union[List[str], str]
        :return: A list of generated images corresponding to the input prompts.
        :rtype: List[Image]
        """
        if isinstance(input, str):
            input = [input]

        images = []
        for prompt in tqdm(input, desc="Images to Generate", leave=False, total=len(input)):
            # Generate images in batches
            image = self.model(
                prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
            ).images[0]

            images.append(image)

        return images
