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
import matplotlib.pyplot as plt
from typing import Any

from CheckEmbed.plotters import PlotOperation

class CheckEmbedPlot(PlotOperation):
    """
    The CheckEmbedPlot class handles the plotting of CheckEmbed data.

    Inherits from the PlotOperation class and implements its abstract methods.
    """

    def __init__(self, result_dir_path: str, data_dir_path: str) -> None:
        """
        Initialize the operation.

        :param result_dir_path: The path to the directory where the results will be stored.
        :type result_dir_path: str
        :param data_dir_path: The path to the directory where the data is stored.
        :type data_dir_path: str
        """
        super().__init__(result_dir_path, data_dir_path)

    def execute(self, custom_inputs: Any = None) -> Any:
        """
        Plot the data.

        :param custom_inputs: The custom inputs for the operation. Defaults to None.
        :type custom_inputs: Any
        """
        print("Running CheckEmbedPlot operation.")
        
        for file in os.listdir(self.data_dir_path):
            if ".json" in file:
                
                folder_name = file.replace("_" + file.split("_")[2], "")

                # Directory creation
                if not os.path.exists(os.path.join(self.result_dir_path, folder_name)):
                    os.mkdir(os.path.join(self.result_dir_path, folder_name))

                if not os.path.exists(os.path.join(self.result_dir_path, folder_name, "cosine_sim")):
                    os.mkdir(os.path.join(self.result_dir_path, folder_name, "cosine_sim"))

                if not os.path.exists(os.path.join(self.result_dir_path, folder_name, "pearson_corr")):
                    os.mkdir(os.path.join(self.result_dir_path, folder_name, "pearson_corr"))

                # Load the results
                with open(os.path.join(self.data_dir_path, file), "r") as f:
                    data = json.load(f)
                data_array = data["data"]
                cosine_similarity_matrix_array = [np.array(d["cosine_sim"]) for d in data_array]  # Convert to numpy array
                frobenius_norms_cosine_sim = [np.array(d["frob_norm_cosine_sim"]) for d in data_array]
                std_dev_cosine_sim_array = [np.array(d["std_dev_cosine_sim"]) for d in data_array]
                pearson_corr_array = [np.array(d["pearson_corr"]) for d in data_array]
                frobenius_norms_pearson_corr = [np.array(d["frob_norm_pearson_corr"]) for d in data_array]
                std_dev_pearson_corr_array = [np.array(d["std_dev_pearson_corr"]) for d in data_array]

                # Plot the heatmap of each cosine_similarity_matrix for every example
                for index, cosine_similarity_matrix in enumerate(cosine_similarity_matrix_array):
                    fig, ax = plt.subplots(figsize=(12, 10))  # Adjust the figure size as needed
                    
                    im = ax.imshow(cosine_similarity_matrix, cmap='YlGnBu', interpolation='nearest', aspect="auto", vmin=-1, vmax=1)
                    plt.colorbar(im, ax=ax)  # Use ax argument to specify the axis for the colorbar

                    plt.title(f"Heatmap of Cosine Similarity of Example {index}", weight='bold', fontsize=26)  # Add a title with index starting from 1
                    plt.xlabel("LLM Reply ID or Ground-Truth (GT)", fontsize=18)
                    plt.ylabel("LLM Reply ID or Ground-Truth (GT)", fontsize=18)

                    # Set ticks and labels
                    tick_labels = list(range(1, len(cosine_similarity_matrix))) + ['GT'] if custom_inputs["ground_truth"] else list(range(1, len(cosine_similarity_matrix) + 1))
                    ax.set_xticks(np.arange(len(cosine_similarity_matrix)))
                    ax.set_yticks(np.arange(len(cosine_similarity_matrix)))
                    ax.set_xticklabels(tick_labels, fontsize=18)
                    ax.set_yticklabels(tick_labels, fontsize=18)

                    # Add numbers to the heatmap
                    for i in range(len(cosine_similarity_matrix)):
                        for j in range(len(cosine_similarity_matrix)):
                            ax.text(j, i, round(cosine_similarity_matrix[i, j], 2), ha="center", va="center", color="red", fontsize=18)
                    
                    plt.savefig(os.path.join(self.result_dir_path, folder_name, "cosine_sim", f"cosine_sim_{index}.pdf"), bbox_inches='tight')
                    plt.close()

                # Plot each Pearson correlation matrix
                for index, pearson_corr_matrix in enumerate(pearson_corr_array):
                    fig, ax = plt.subplots(figsize=(12, 10))  # Adjust the figure size as needed
                    
                    im = ax.imshow(pearson_corr_matrix, cmap='YlGnBu', interpolation='nearest', aspect="auto", vmin=0, vmax=1)
                    plt.colorbar(im, ax=ax)  # Use ax argument to specify the axis for the colorbar

                    plt.title(f"Heat Map of Pearson Correlation of Example {index}")  # Add a title with index starting from 1
                    plt.xlabel("LLM Reply ID or Ground-Truth (GT)", fontsize=18)
                    plt.ylabel("LLM Reply ID or Ground-Truth (GT)", fontsize=18)

                    # Set ticks and labels
                    tick_labels = list(range(1, len(pearson_corr_matrix))) + ['GT'] if custom_inputs["ground_truth"] else list(range(1, len(pearson_corr_matrix) + 1))
                    ax.set_xticks(np.arange(len(pearson_corr_matrix)))
                    ax.set_yticks(np.arange(len(pearson_corr_matrix)))
                    ax.set_xticklabels(tick_labels, fontsize=18)
                    ax.set_yticklabels(tick_labels, fontsize=18)

                    # Add numbers to the heatmap
                    for i in range(len(pearson_corr_matrix)):
                        for j in range(len(pearson_corr_matrix)):
                            ax.text(j, i, round(pearson_corr_matrix[i, j], 2), ha="center", va="center", color="red", fontsize=18)
                    
                    plt.savefig(os.path.join(self.result_dir_path, folder_name, "pearson_corr", f"pearson_corr_{index}.pdf"), bbox_inches='tight')
                    plt.close()
                
                # Plot the Frobenius norm of the cosine similarity matrices
                fig, ax = plt.subplots()
                ax.bar(range(len(frobenius_norms_cosine_sim)), frobenius_norms_cosine_sim)
                ax.set_xlabel("Prompt")
                ax.set_ylabel("Frobenius Norm")
                ax.set_title("Frobenius Norm of Cosine Similarity Matrices")

                tick_labels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                ax.set_yticks(tick_labels)
                ax.set_yticklabels(tick_labels)
                tick_labels = list(range(1, len(frobenius_norms_cosine_sim) + 1))
                ax.set_xticks(np.arange(len(frobenius_norms_cosine_sim)))
                ax.set_xticklabels(tick_labels)

                plt.savefig(os.path.join(self.result_dir_path, folder_name, "frobenius_norms_cosine_sim.pdf"), bbox_inches='tight')
                plt.close()

                # Plot the Frobenius norm of the Pearson correlation matrices
                fig, ax = plt.subplots()
                ax.bar(range(len(frobenius_norms_pearson_corr)), frobenius_norms_pearson_corr)
                ax.set_xlabel("Prompt")
                ax.set_ylabel("Frobenius Norm")
                ax.set_title("Frobenius Norm of Pearson Correlation Matrices")

                tick_labels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                ax.set_yticks(tick_labels)
                ax.set_yticklabels(tick_labels)
                tick_labels = list(range(1, len(frobenius_norms_pearson_corr) + 1))
                ax.set_xticks(np.arange(len(frobenius_norms_pearson_corr)))
                ax.set_xticklabels(tick_labels)

                plt.savefig(os.path.join(self.result_dir_path, folder_name, "frobenius_norms_pearson_corr.pdf"), bbox_inches='tight')
                plt.close()

                # Plot the standard deviation of the cosine similarity matrices
                fig, ax = plt.subplots()
                ax.bar(range(len(std_dev_cosine_sim_array)), std_dev_cosine_sim_array)
                ax.set_xlabel("Prompt")
                ax.set_ylabel("Standard Deviation")
                ax.set_title("Standard Deviation of Cosine Similarity Matrices")

                # Set min to 0 and max to 0.3 for the y-axis
                tick_labels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                ax.set_yticks(tick_labels)
                ax.set_yticklabels(tick_labels)
                tick_labels = list(range(1, len(std_dev_cosine_sim_array) + 1))
                ax.set_xticks(np.arange(len(std_dev_cosine_sim_array)))
                ax.set_xticklabels(tick_labels)

                plt.savefig(os.path.join(self.result_dir_path, folder_name, "std_dev_cosine_sim.pdf"), bbox_inches='tight')
                plt.close()

                # Plot the standard deviation of the Pearson correlation matrices
                fig, ax = plt.subplots()
                ax.bar(range(len(std_dev_pearson_corr_array)), std_dev_pearson_corr_array)
                ax.set_xlabel("Prompt")
                ax.set_ylabel("Standard Deviation")
                ax.set_title("Standard Deviation of Pearson Correlation Matrices")

                # Set min to 0 and max to 0.3 for the y-axis
                tick_labels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                ax.set_yticks(tick_labels)
                ax.set_yticklabels(tick_labels)
                tick_labels = list(range(1, len(std_dev_pearson_corr_array) + 1))
                ax.set_xticks(np.arange(len(std_dev_pearson_corr_array)))
                ax.set_xticklabels(tick_labels)

                plt.savefig(os.path.join(self.result_dir_path, folder_name, "std_dev_pearson_corr.pdf"), bbox_inches='tight')
                plt.close()
