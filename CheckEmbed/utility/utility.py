# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import numpy as np
import math

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        :param a: The first vector.
        :type a: np.ndarray
        :param b: The second vector.
        :type b: np.ndarray
        :return: The cosine similarity between the two vectors.
        :rtype: float
        """

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
        return cos_similarity

def frobenius_norm(matrix: np.ndarray) -> float:
    """
    Compute the Frobenius norm of the input matrix normalized by the number of elements in the matrix.

    :param matrix: Input matrix.
    :type matrix: np.ndarray
    :return: Frobenius norm.
    :rtype: float
    """
    sum = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            sum += matrix[i, j] ** 2
    
    # normalize by the number of elements in the matrix
    return math.sqrt(sum / (matrix.shape[0] * matrix.shape[1]))

def frobenius_norm_no_diag(matrix: np.ndarray) -> float:
    """
    Compute the Frobenius norm of the input matrix without its diagonal elements.
    The Frobenius is further normalized by the number of elements in the matrix.

    :param matrix: Input matrix.
    :type matrix: np.ndarray
    :return: Frobenius norm.
    :rtype: float
    """
    matrix_no_diag = matrix[~np.eye(matrix.shape[0],dtype=bool)].reshape(matrix.shape[0],-1)
    return frobenius_norm(matrix_no_diag)

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
