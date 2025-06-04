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
from PIL.Image import Image
from transformers import CLIPModel, CLIPProcessor

from CheckEmbed.embedding_models import AbstractEmbeddingModel


class ClipVitLarge(AbstractEmbeddingModel):
    """
    The ClipVitLarge class handles interactions with the CLIP ViT Large model using the provided configuration.

    Inherits from the AbstractEmbeddingModel class and implements its abstract methods.
    """

    def __init__(
        self, model_name: str = "", name: str = "clip-vit-large-patch-14", cache: bool = False
    ) -> None:
        """
        Initialize the ClipVitLarge instance with configuration, model details, and caching options.
        :param model_name: Name of the model, which is used to select the correct configuration. Defaults to "".
        :type model_name: str
        :param name: Name used for output files. Defaults to "clip-vit-large-patch-14".
        :type name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        super().__init__(model_name=model_name, name=name, cache=cache)
        self.processor_name = model_name

    def load_model(self, device: str = None) -> None:
        """
        Load the model and tokenizer based on the given model name.

        :param device: The device to load the model on. Defaults to None.
        :type device: str
        """
        self.model = CLIPModel.from_pretrained(self.model_name).eval()
        self.processor = CLIPProcessor.from_pretrained(self.processor_name)
        self.model = self.model.to(device)

    def unload_model(self) -> None:
        """
        Unload the model and tokenizer.
        """
        del self.processor
        del self.model

        gc.collect()
        torch.cuda.empty_cache()

        self.processor = None
        self.model = None

    def generate_embedding(self, input: Union[List[Image], Image]) -> List[List[float]]:
        """
        Abstract method to generate embedding for the given input text.

        :param input: The input image to embed.
        :type input: Union[List[Image], Image]
        :return: The embeddings of the image.
        :rtype: List[List[float]]
        """
        if not isinstance(input, List):
            input = [input]

        total_embeddings = []
        for image in input:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                latents = self.model.get_image_features(**inputs).squeeze().cpu().numpy().tolist()
            total_embeddings.append(latents)
        return total_embeddings
