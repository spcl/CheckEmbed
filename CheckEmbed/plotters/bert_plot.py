# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import os
import json

import matplotlib.pyplot as plt
import numpy as np
from typing import Any

from CheckEmbed.plotters import PlotOperation

class BertPlot(PlotOperation):
    """
    The BertPlot class handles the plotting of BERTScore data.

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
        print("Running BertPlot operation.")
        
        for file in os.listdir(self.data_dir_path):
            if ".json" in file:

                if not os.path.exists(os.path.join(self.result_dir_path, file.split("_")[0])):
                    os.mkdir(os.path.join(self.result_dir_path, file.split("_")[0]))

                with open(os.path.join(self.data_dir_path, file), "r") as f:
                    data = json.load(f)

                data_array = data["data"]
                results = [np.array(d["result"]) for d in data_array]
                frobenius_norms = [np.array(d["frobenius_norm"]) for d in data_array]

                # Plot a separate heatmap for every example
                for index, result in enumerate(results):
                    fig, ax = plt.subplots(figsize=(12, 10))  # Adjust the figure size as needed
                    
                    im = ax.imshow(result, cmap='YlGnBu', interpolation='nearest', aspect="auto", vmin=-1, vmax=1)
                    plt.colorbar(im, ax=ax)  # Use ax argument to specify the axis for the colorbar

                    plt.title(f"Heatmap of BertScore of Example {index}", weight='bold', fontsize=26)  # Add a title with index starting from 1
                    plt.xlabel("LLM Reply ID or Ground-Truth (GT)", fontsize=18)
                    plt.ylabel("LLM Reply ID or Ground-Truth (GT)", fontsize=18)

                    # Set ticks and labels
                    tick_labels = list(range(1, result.shape[0])) + ['GT'] if custom_inputs["ground_truth"] else list(range(1, result.shape[0] + 1))
                    ax.set_xticks(np.arange(result.shape[0]))
                    ax.set_yticks(np.arange(result.shape[0]))
                    ax.set_xticklabels(tick_labels, fontsize=18)
                    ax.set_yticklabels(tick_labels, fontsize=18)

                    # Add numbers to the heatmap
                    for i in range(result.shape[0]):
                        for j in range(result.shape[0]):
                            text = ax.text(j, i, round(result[i, j], 2), ha="center", va="center", color="red", fontsize=18)
                    
                    plt.savefig(os.path.join(self.result_dir_path, file.split("_")[0], f"example_{index}.pdf"), bbox_inches='tight')
                    plt.close()

                # Plot the Frobenius norm of the cosine similarity matrices
                fig, ax = plt.subplots()
                ax.bar(range(len(frobenius_norms)), frobenius_norms)
                ax.set_xlabel("Prompt")
                ax.set_ylabel("Frobenius Norm")
                ax.set_title("Frobenius Norm of BertScore Matrices")

                tick_labels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                ax.set_yticks(tick_labels)
                ax.set_yticklabels(tick_labels)
                tick_labels = list(range(1, len(frobenius_norms) + 1))
                ax.set_xticks(np.arange(len(frobenius_norms)))
                ax.set_xticklabels(tick_labels)

                plt.savefig(os.path.join(self.result_dir_path, file.split("_")[0], "frobenius_norm.pdf"), bbox_inches='tight')
                plt.close()
