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


import asyncio
import backoff
import os
from typing import List, Dict, Union
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat.chat_completion import ChatCompletion
from tqdm import tqdm

from CheckEmbed.language_models import AbstractLanguageModel


class ChatGPT(AbstractLanguageModel):
    """
    The ChatGPT class handles interactions with the OpenAI models using the provided configuration.

    Inherits from the AbstractLanguageModel class and implements its abstract methods.
    """

    # modified by Lorenzo Paleari
    def __init__(
        self, config_path: str = "", model_name: str = "chatgpt4", cache: bool = False, max_concurrent_requests: int = 10
    ) -> None:
        """
        Initialize the ChatGPT instance with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the model, default is 'chatgpt4'. Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        :param max_concurrent_requests: The maximum number of concurrent requests. Defaults to 10.
        :type max_concurrent_requests: int
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        # The model_id is the id of the model that is used for chatgpt, i.e. gpt-4, gpt-3.5-turbo, etc.
        self.model_id: str = self.config["model_id"]
        # The prompt_token_cost and response_token_cost are the costs for 1000 prompt tokens and 1000 response tokens respectively.
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        # The temperature of a model is defined as the randomness of the model's output.
        self.temperature: float = self.config["temperature"]
        # The maximum number of tokens to generate in the chat completion.
        self.max_tokens: int = self.config["max_tokens"]
        # The stop sequence is a sequence of tokens that the model will stop generating at (it will not generate the stop sequence).
        self.stop: Union[str, List[str]] = self.config["stop"]
        # The account organization is the organization that is used for chatgpt.
        self.organization: str = self.config["organization"]
        if self.organization == "":
            self.logger.warning("OPENAI_ORGANIZATION is not set")
        self.api_key: str = os.getenv("OPENAI_API_KEY", self.config["api_key"])
        if self.api_key == "":
            self.logger.warning("OPENAI_API_KEY is not set")
        # Initialize the OpenAI Client
        self.client = AsyncOpenAI(api_key=self.api_key, organization=self.organization)

        self.max_concurrent_requests = max_concurrent_requests

    # written by Lorenzo Paleari
    def load_model(self, device: str = None) -> None:
        """
        Load the language model locally.

        :param device: The device to load the model on.
        :type device: str
        """
        pass

    # written by Lorenzo Paleari
    def unload_model(self) -> None:
        """
        Unload the language model locally.
        """
        pass

    # modified by Lorenzo Paleari
    def query(
        self, query: str, num_query: int = 1
    ) -> List[ChatCompletion]:
        """
        Query the OpenAI model for responses.

        :param query: The prompt that is going to be used as query to the language model.
        :type query: str
        :param num_query: The number of queries to be posed to the language model for each prompt. Defaults to 1.
        :type num_query: int
        :return: Response(s) from the OpenAI model.
        :rtype: List[ChatCompletion]
        """
        async def async_query() -> List[ChatCompletion]:
            if self.cache and query in self.response_cache:
                self.logger.debug(f"Used cache for query: {query}")
                return self.response_cache[query]
            
            async def sem_task(semaphore, task):
                async with semaphore:
                    return await task

            semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            tasks = [sem_task(semaphore, self.chat([{"role": "user", "content": query}], 1)) for _ in range(num_query)]
            
            responses = []
            for task in tqdm(asyncio.as_completed(tasks), total=num_query, desc="Samples", leave=False):
                response = await task
                responses.append(response)

            if self.cache:
                self.response_cache[query] = response
            return responses
        
        return asyncio.run(async_query())

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    async def chat(self, messages: List[Dict], num_responses: int = 1) -> ChatCompletion:
        """
        Send chat messages to the OpenAI model and retrieves the model's response.
        Implements backoff on OpenAI error.

        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: The OpenAI model's response.
        :rtype: ChatCompletion
        """
        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=num_responses,
            stop=self.stop,
        )

        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        prompt_tokens_k = float(self.prompt_tokens) / 1000.0
        completion_tokens_k = float(self.completion_tokens) / 1000.0
        self.cost = (
            self.prompt_token_cost * prompt_tokens_k
            + self.response_token_cost * completion_tokens_k
        )
        self.logger.info(
            f"This is the response from chatgpt: {response}"
            f"\nThis is the cost of the response: {self.cost}"
        )
        return response

    def get_response_texts(
        self, query_response: Union[List[ChatCompletion], ChatCompletion]
    ) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_response: The response dictionary (or list of dictionaries) from the OpenAI model.
        :type query_response: Union[List[ChatCompletion], ChatCompletion]
        :return: List of response strings.
        :rtype: List[str]
        """
        if not isinstance(query_response, List):
            query_response = [query_response]
        return [
            choice.message.content
            for response in query_response
            for choice in response.choices
        ]
