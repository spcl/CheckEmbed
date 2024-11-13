# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import numpy as np
import math

# These values are the lowest empiricial values observed for a given
# embedding model during our evaluation.
REBASING_VALUES = {
    "gpt-embedding-large": 0.36156142737003805,
    "sfr-embedding-mistral": 0.4590856938212389,
    "e5-mistral-7B-instruct": 0.5347691513588488,
    "gte-qwen1.5-7B-instruct": 0.17701296296393593,
    "stella-en-400M-v5": 0.3189337589450308,
    "stella-en-1.5B-v5": 0.3655769126487221,
}

def cosine_similarity(a: np.ndarray, b: np.ndarray, rebase: bool = False, emb_name: str = "") -> float:
        """
        Compute cosine similarity between two vectors.

        :param a: The first vector.
        :type a: np.ndarray
        :param b: The second vector.
        :type b: np.ndarray
        :param rebase: Whether to rebase the cosine similarity. Defaults to False.
        :type rebase: bool
        :param emb_name: The name of the embedding model. Defaults to "".
        :type emb_name: str
        :return: The cosine similarity between the two vectors.
        :rtype: float
        """
        global REBASING_VALUES

        # Special case for empty vectors
        if len(a) == 0 and len(b) == 0:
            return 1.0
        if len(a) == 0 or len(b) == 0:
            return -1.0
        
        # Compute the cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cos_similarity = dot_product / (norm_a * norm_b)

        if rebase and emb_name in REBASING_VALUES:
            # Rebase the cosine similarity
            cos_similarity = 2 * (cos_similarity - REBASING_VALUES[emb_name]) / (1.0 - REBASING_VALUES[emb_name]) - 1.0
            cos_similarity = 1.0 if cos_similarity > 1.0 else -1.0 if cos_similarity < -1.0 else cos_similarity

        return cos_similarity

def frobenius_norm(matrix: np.ndarray, bert: bool = False) -> float:
    """
    Compute the Frobenius norm of the input matrix normalized by the number of elements in the matrix.

    :param matrix: Input matrix.
    :type matrix: np.ndarray
    :param bert: Whether the matrix is a BertScore matrix. Defaults to False.
    :type bert: bool
    :return: Frobenius norm.
    :rtype: float
    """
    adder = 1
    div = 4
    if bert:
        adder = 0
        div = 1
    sum = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            sum += (matrix[i, j] + adder) ** 2
    
    # normalize by the number of elements in the matrix
    return math.sqrt(sum / (matrix.shape[0] * matrix.shape[1] * div))

def frobenius_norm_no_diag(matrix: np.ndarray, bert: bool = False) -> float:
    """
    Compute the Frobenius norm of the input matrix without its diagonal elements.
    The Frobenius is further normalized by the number of elements in the matrix.

    :param matrix: Input matrix.
    :type matrix: np.ndarray
    :param bert: Whether the matrix is a BertScore matrix. Defaults to False.
    :type bert: bool
    :return: Frobenius norm.
    :rtype: float
    """
    matrix_no_diag = matrix[~np.eye(matrix.shape[0],dtype=bool)].reshape(matrix.shape[0],-1)
    return frobenius_norm(matrix_no_diag, bert)

def matrix_std_dev_no_diag(matrix: np.ndarray) -> float:
    """
    Compute the standard deviation of the input matrix without its diagonal elements.

    :param matrix: Input matrix.
    :type matrix: np.ndarray
    :return: Standard deviation.
    :rtype: float
    """
    matrix_no_diag = matrix[~np.eye(matrix.shape[0],dtype=bool)].reshape(matrix.shape[0],-1)
    return np.std(matrix_no_diag)
