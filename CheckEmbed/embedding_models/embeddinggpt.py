# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import asyncio
import backoff
import os
from typing import List, Dict, Union
from openai import AsyncOpenAI, OpenAIError
from openai.types import CreateEmbeddingResponse
from tqdm import tqdm

from CheckEmbed.embedding_models import AbstractEmbeddingModel


class EmbeddingGPT(AbstractEmbeddingModel):
    """
    The EmbeddingGPT class handles interactions with the OpenAI embedding models using the provided configuration.

    Inherits from the AbstractEmbeddingModel class and implements its abstract methods.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "chatgpt4", cache: bool = False, max_concurrent_requests: int = 10
    ) -> None:
        """
        Initialize the EmbeddingGPT instance with configuration, model details, and caching options.

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
        self.prompt_token_cost: float = self.config["token_cost"]
        self.encoding: str = self.config["encoding"]
        self.dimension: int = self.config["dimension"]
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

    def load_model(self, device: str = None) -> None:
        """
        Load the embedding model locally.

        :param device: The device to load the model on.
        :type device: str
        """
        pass

    def unload_model(self) -> None:
        """
        Unload the embedding model locally.
        """
        pass

    def generate_embedding(self, input: Union[List[str], str]) -> List[List[float]]:
        """
        Abstract method to generate embedding for the given input text.

        :param input: The input texts to embed.
        :type input: Union[List[str], str]
        :return: The embeddings of the text.
        :rtype: List[List[float]]
        """
        async def async_query(input: Union[List[str], str]):
            if isinstance(input, str):
                input = [input]
            
            async def sem_task(semaphore, task):
                async with semaphore:
                    return await task
                
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            tasks = [sem_task(semaphore, self.embed_query(i)) for i in input]

            responses = []
            for task in tqdm(asyncio.as_completed(tasks), total=len(input), desc="Embeddings", leave=False):
                response = await task
                responses.append(response.data[0].embedding)

            return responses
        
        return asyncio.run(async_query(input))

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    async def embed_query(self, input: str) -> CreateEmbeddingResponse:
        """
        Embed the given text into a vector.

        :param input: The text to embed.
        :type input: str
        :return: The embedding of the text.
        :rtype: CreateEmbeddingResponse
        """
        response = await self.client.embeddings.create(
            model=self.model_id,
            input=input,
            dimensions=self.dimension,
            encoding_format=self.encoding,
        )

        self.prompt_tokens += response.usage.prompt_tokens
        prompt_tokens_k = float(self.prompt_tokens) / 1000.0
        self.cost = (
            self.prompt_token_cost * prompt_tokens_k
        )
        self.logger.info(
            f"This is the response from chatgpt: {response}"
            f"\nThis is the cost of the response: {self.cost}"
        )
        return response
