# Copyright (c) 2023, 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# original file from Graph of Thoughts framework:
# https://github.com/spcl/graph-of-thoughts
#
# main author: Nils Blach
#
# modifications: Lorenzo Paleari

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import json
import logging


class AbstractLanguageModel(ABC):
    """
    Abstract base class that defines the interface for all language models.
    """

    # modified by Lorenzo Paleari
    def __init__(
        self, config_path: str = None, model_name: str = "", cache: bool = False
    ) -> None:
        """
        Initialize the AbstractLanguageModel instance with configuration, model details, and caching options.

        :param config_path: Path to the config file. Defaults to None. If provided, the config is loaded from the file.
        :type config_path: str
        :param model_name: Name of the language model. Defaults to "".
        :type model_name: str
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
        self.name: str = self.config[model_name]["name"]
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.cost: float = 0.0

    # modified by Lorenzo Paleari
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

    # written by Lorenzo Paleari
    @abstractmethod
    def load_model(self, device: str = None) -> None:
        """
        Abstract method to load the language model.

        :param device: The device to load the model on.
        :type device: str
        """
        pass

    # written by Lorenzo Paleari
    @abstractmethod
    def unload_model(self) -> None:
        """
        Abstract method to unload the language model.
        """
        pass

    # modified by Lorenzo Paleari
    @abstractmethod
    def query(self, query: str, num_query: int = 1) -> Any:
        """
        Abstract method to query the language model.

        :param query: The prompt that is going to be used as query to the language model.
        :type query: str
        :param num_query: The number of queries to be posed to the language model for each prompt. Defaults to 1.
        :type num_query: int
        :return: The language model's response(s).
        :rtype: Any
        """
        pass
