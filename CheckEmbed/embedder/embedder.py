# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

from abc import ABC
from typing import List

from tqdm import tqdm
from CheckEmbed.embedding_models import AbstractEmbeddingModel
import numpy as np

class Embedder(ABC):
    """
    Abstract base class that defines the interface for all embedders.
    Embedders are used to embed text into a vector space.
    """

    def embed(self, lm: AbstractEmbeddingModel, texts: List[str]) -> List[List[float]]:
        """
        Embed the given texts into vectors.

        :param lm: The embedding model that will be used to generate the text embeddings.
        :type lm: AbstractEmbeddingModel
        :param texts: The texts to embed.
        :type texts: List[str]
        :return: The embeddings of the texts.
        :rtype: List[List[float]]
        """
        embedding_query = []
        void_indexes = []
        for index, text in enumerate(texts):
            if text == "":
                void_indexes.append(index)
            else:
                embedding_query.append(text)

        full_responses = np.zeros((len(texts))).tolist()
        responses = lm.generate_embedding(embedding_query)

        for index in void_indexes:
            full_responses[index] = []

        # fill remaining places in full_responses with responses in oroder
        for index, response in enumerate(responses):
            temp_index = index
            while full_responses[temp_index] != 0.0:
                temp_index += 1
            full_responses[temp_index] = response
            
        return full_responses
