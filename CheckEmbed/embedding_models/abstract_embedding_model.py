# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import json
import logging


class AbstractEmbeddingModel(ABC):
    """
    Abstract base class that defines the interface for all embedding models.
    """

    def __init__(
        self, config_path: str = None, model_name: str = "", name: str = "INVALID_NAME", cache: bool = False
    ) -> None:
        """
        Initialize the AbstractEmbeddingModel instance with configuration, model details, and caching options.

        :param config_path: Path to the config file. If provided, the config is loaded from the file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the language model. Defaults to "".
        :type model_name: str
        :param name: Name of the embedding model. Defaults to "INVALID_NAME".
        :type name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config: Dict = None
        self.model_name: str = model_name
        self.cache = cache
        if self.cache:
            self.response_cache: Dict[str, List[Any]] = {}
        if config_path is not None:
            self.load_config(config_path)
        self.name: str = name
        try: 
            if self.config is not None:
                if self.config[model_name] is not None:
                    self.name = self.config[model_name]["name"]
        except Exception:
            pass
        self.prompt_tokens: int = 0
        self.cost: float = 0.0

    def load_config(self, path: str) -> None:
        """
        Load configuration from a specified path.

        :param path: Path to the config file.
        :type path: str
        """
        with open(path, "r") as f:
            self.config = json.load(f)

        self.logger.debug(f"Loaded config from {path} for {self.model_name}")

    def clear_cache(self) -> None:
        """
        Clear the response cache.
        """
        self.response_cache.clear()

    @abstractmethod
    def load_model(self, device: str = None) -> None:
        """
        Abstract method to load the embedding model.

        :param device: The device to load the model on. Defaults to None.
        :type device: str
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """
        Abstract method to unload the embedding model.
        """
        pass

    @abstractmethod
    def generate_embedding(self, input: Union[List[Any], Any]) -> List[List[float]]:
        """
        Abstract method to generate embedding for the given input text.

        :param input: The input text to embed.
        :type input: Union[List[Any], Any]
        :return: The embeddings of the text.
        :rtype: List[List[float]]
        """
        pass
