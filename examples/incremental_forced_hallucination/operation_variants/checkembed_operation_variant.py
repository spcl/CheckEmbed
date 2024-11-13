# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import os
import json

import numpy as np

from typing import Any

from CheckEmbed.operations import Operation
from CheckEmbed.utility import cosine_similarity, frobenius_norm

class CheckEmbedOperation_Variant(Operation):
    """
    Operation that computes the cosine similarity, the Pearson correlation, the Frobenius norm and standard deviation between the embeddings.
    """

    def __init__(self, result_dir_path: str, ground_truth_dir_path: str, embeddings_dir_path: str) -> None:
        """
        Initialize the operation.

        :param result_dir_path: The path to the directory where the results will be stored.
        :type result_dir_path: str
        :param ground_truth_dir_path: The path to the directory where the ground truth embeddings are stored.
        :type ground_truth_dir_path: str
        :param embeddings_dir_path: The path to the directory where the embeddings are stored.
        :type embeddings_dir_path: str
        """
        super().__init__(result_dir_path)
        self.ground_truth_dir = ground_truth_dir_path
        self.embeddings_dir_path = embeddings_dir_path

    def execute(self, custom_inputs: Any) -> Any:
        """
        Execute the operation on the embeddings/samples.

        :param custom_inputs: The custom inputs for the operation.
        :type custom_inputs: Any
        """

        # For every language model / embedding model 
        for file in os.listdir(self.embeddings_dir_path):
            if ".json" in file and not file.startswith("ground_truth_"):
                
                folder_name = file.replace("_" + file.split("_")[2], "")

                # Load the samples embeddings
                with open(os.path.join(self.embeddings_dir_path, file), "r") as f:
                    data = json.load(f)
                data_array = data["data"]
                embeddings = [d["embeddings"] for d in data_array]  # Convert to numpy array

                # Load the ground_truth embeddings
                dimensions = len(embeddings[0])
                with open(os.path.join(self.ground_truth_dir, file), "r") as f:
                    data = json.load(f)
                data_array = data["data"]
                ground_truth_embeddings = [d["embeddings"] for d in data_array]  # Convert to numpy array

                # Compute the cosine similarity matrix with the ground truth
                cosine_similarity_matrix_array = []
                for embedding, ground_truth in zip(embeddings, ground_truth_embeddings):
                    # -1 array to initialize the cosine similarity matrix  
                    cosine_similarity_matrix = np.full((dimensions, dimensions), -1.0)
                    for i in range(len(embedding)):
                        for j in range(len(embedding)):
                            cosine_similarity_matrix[i, j] = cosine_similarity(ground_truth[i], embedding[j])

                    cosine_similarity_matrix_array.append(cosine_similarity_matrix)
        
                # Compute the Frobenius norm of each cosine similarity matrix
                frobenius_norms_cosine_sim = [frobenius_norm(cosine_similarity_matrix) for cosine_similarity_matrix in cosine_similarity_matrix_array]

                # Compute the standard deviation of each cosine similarity matrix
                std_dev_cosine_sim_array = [np.std(cosine_similarity_matrix) for cosine_similarity_matrix in cosine_similarity_matrix_array]

                # Compute the Pearson correlation matrix
                pearson_corr_array = []
                for embedding, ground_truth in zip(embeddings, ground_truth_embeddings):
                    pearson_corr = np.full((dimensions, dimensions), -1.0)
                    for i in range(len(embedding)):
                        for j in range(len(embedding)):
                            if len(embedding[i]) == 0 and len(embedding[j]) == 0:
                                pearson_corr[i, j] = 1.0
                                continue
                            if len(embedding[i]) == 0 or len(embedding[j]) == 0:
                                pearson_corr[i, j] = -1.0
                                continue
                            pearson_corr[i, j] = np.corrcoef(embedding[i], embedding[j])[0, 1]

                    pearson_corr_array.append(pearson_corr)

                # Compute the Frobenius norm of each cosine similarity matrix
                frobenius_norms_pearson_corr = [frobenius_norm(pearson_corr) for pearson_corr in pearson_corr_array]

                std_dev_pearson_corr_array = [np.std(pearson_corr) for pearson_corr in pearson_corr_array]

                with open(os.path.join(self.result_dir_path, folder_name + "_results.json"), "w") as f:
                    results_json = [{
                        "index": index,
                        "cosine_sim": cosine_sim.tolist(),
                        "frob_norm_cosine_sim": frob_norm_cosine_sim,
                        "std_dev_cosine_sim": std_dev_cosine_sim,
                        "pearson_corr": pearson_corr.tolist(),
                        "frob_norm_pearson_corr": frob_norm_pearson_corr,
                        "std_dev_pearson_corr": std_dev_pearson_corr
                    } for index, cosine_sim, frob_norm_cosine_sim, std_dev_cosine_sim, pearson_corr, frob_norm_pearson_corr, std_dev_pearson_corr in zip(range(len(cosine_similarity_matrix_array)), cosine_similarity_matrix_array, frobenius_norms_cosine_sim, std_dev_cosine_sim_array, pearson_corr_array, frobenius_norms_pearson_corr, std_dev_pearson_corr_array)]
                    json.dump({"data": results_json}, f, indent=4)
