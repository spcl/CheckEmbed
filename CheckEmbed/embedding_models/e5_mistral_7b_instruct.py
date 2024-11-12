# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import torch
import gc
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from typing import List, Union

from CheckEmbed.embedding_models import AbstractEmbeddingModel


class E5Mistral7b(AbstractEmbeddingModel):
    """
    The E5Mistral7b class handles interactions with the E5Mistral7b embedding model using the provided configuration.

    Inherits from the AbstractEmbeddingModel class and implements its abstract methods.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "", cache: bool = False, max_length: int = 4096, batch_size: int = 64
    ) -> None:
        """
        Initialize the E5Mistral7b instance with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the model, default is "". Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        :param max_length: The maximum length of the input text.
        :type max_length: int
        :param batch_size: The batch size to be used for the model.
        :type batch_size: int
        """
        super().__init__(config_path, model_name, cache)
        self.tokenizer_name = model_name
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

    def load_model(self, device: str = None) -> None:
        """
        Load the model and tokenizer based on the given model name.

        :param device: The device to load the model on.
        :type device: str
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModel.from_pretrained(self.model_name, device_map=device)
    
    def unload_model(self) -> None:
        """
        Unload the model and tokenizer.
        """
        del self.tokenizer
        del self.model

        gc.collect()
        torch.cuda.empty_cache()

        self.tokenizer = None
        self.model = None

    def generate_embedding(self, input: Union[List[str], str]) -> List[List[float]]:
        """
        Abstract method to generate embedding for the given input text.

        :param input: The input text to embed.
        :type input: Union[List[str], str]
        :return: The embeddings of the text.
        :rtype: List[List[float]]
        """
        if isinstance(input, str):
            input = [input]

        total_embeddings = []
        flag = True

        while flag:
            try:
                batched_responses = [input[i:i+self.batch_size] for i in range(0, len(input), self.batch_size)]

                embeddings = None
                outputs = None
                batch_dict = None

                for batch in tqdm(batched_responses, desc="Batches to Embed", leave=False, total=len(batched_responses)):
                    batch_dict = self.tokenizer(batch, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
                    batch_dict.to(self.model.device)

                    with torch.no_grad():
                        outputs = self.model(**batch_dict)
                    embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

                    # normalize embeddings
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    total_embeddings.extend(embeddings.cpu().detach().numpy().tolist())

                    del embeddings, outputs, batch_dict
                    gc.collect()
                    torch.cuda.empty_cache()
                
                flag = False

            except Exception as e:
                embeddings = None
                outputs = None
                batch_dict = None
                total_embeddings = []
                gc.collect()
                torch.cuda.empty_cache()
                
                print("Error occurred, reducing batch size and retrying")
                if self.batch_size == 1:
                    raise e
                self.batch_size = self.batch_size // 2  # reduce batch size by half
            
        return total_embeddings

    def last_token_pool(self, last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        """
        Pools the last non-padding token's hidden state from the model's output.

        This method extracts the hidden state of the last token that is not a padding token.
        If the last token is a padding token, it retrieves the hidden state of the 
        second to last token that is not a padding token.

        :param last_hidden_states: A tensor containing the hidden states from the last layer of the model.
        :type last_hidden_states: Tensor
        :param attention_mask: A tensor indicating the positions of non-padding tokens (1 for non-padding, 0 for padding).
        :type attention_mask: Tensor
        :return: A tensor containing the hidden states of the last non-padding token for each sequence in the batch.
        :rtype: Tensor
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
