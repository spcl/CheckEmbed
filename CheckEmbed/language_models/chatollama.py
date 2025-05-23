# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari


from typing import Dict, List, Union

from langchain_ollama import ChatOllama
from pydantic import BaseModel

from CheckEmbed.language_models import AbstractLanguageModel


class LLMChatOllama(AbstractLanguageModel):
    """
    The LLMChatOllama class handles interactions with Ollama models using the provided configuration.

    Inherits from the AbstractLanguageModel class and implements its abstract methods.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "llama8b", cache: bool = False, temperature: float = None
    ) -> None:
        """
        Initialize the LLMChatOllama instance with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the model, default is 'llama8b'. Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        :param temperature: The temperature for the model. If not provided, it will be taken from the config.
        :type temperature: float
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        self.model_id: str = self.config["model_id"]
        self.name = self.config["name"]
        self.num_ctx = self.config["num_ctx"]
        self.num_predict = self.config["num_predict"]
        self.num_batch = self.config["num_batch"]
        self.keep_alive = self.config["keep_alive"]
        self.temperature: float = temperature if temperature is not None else self.config["temperature"]
        # Initialize the Ollama Client
        self.client = ChatOllama(
            model=self.model_id,
            temperature=self.temperature,
            base_url="localhost:11434",
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
            num_batch=self.num_batch,
            keep_alive=self.keep_alive,
        )

    def load_model(self, device: str = None) -> None:
        """
        Load the language model locally.

        :param device: The device to load the model on.
        :type device: str
        """
        pass

    def unload_model(self) -> None:
        """
        Unload the language model locally.
        """
        pass

    def add_structured_output(self, response: BaseModel) -> None:
        """
        Add structured output to the response.

        :param response: The response from the language model.
        :type response: BaseModel
        """
        self.client = self.client.with_structured_output(
            response, method="json_schema"
        )

    def query(
        self, query: str, num_query: int = 1
    ) -> str:
        """
        Query the Ollama model for responses.

        :param query: The prompt that is going to be used as query to the language model.
        :type query: str
        :param num_query: The number of queries to be posed to the language model for each prompt. Defaults to 1.
        :type num_query: int
        :return: Response(s) from the Ollama model.
        :rtype: str
        """
        if self.cache and query in self.response_cache:
                self.logger.debug(f"Used cache for query: {query}")
                return self.response_cache[query]
    
        result = self.client.invoke(
             query
        )        
        
        if self.cache:
            self.response_cache[query] = result
        return result


    def get_response_texts(
        self, query_response: Union[List[str], str]
    ) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_response: The response dictionary (or list of dictionaries) from the Ollama model.
        :type query_response: Union[List[ChatCompletion], ChatCompletion]
        :return: List of response strings.
        :rtype: List[str]
        """
        pass
