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
from typing import Any

from CheckEmbed.plotters import PlotOperation

class SelfCheckGPTPlot(PlotOperation):
    """
    The SelfCheckGPTPlot class handles the plotting of SelfCheckGPT data.

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
        print("Running SelfCheckGPTPlot operation.")
        
        for file in os.listdir(self.data_dir_path):
            if ".json" in file:

                if not os.path.exists(os.path.join(self.result_dir_path, file.split("_")[0])):
                    os.mkdir(os.path.join(self.result_dir_path, file.split("_")[0]))
                
                with open(os.path.join(self.data_dir_path, file), "r") as f:
                    data = json.load(f)

                data_array = data["data"]
                results = [d["result"] for d in data_array]
                passage_scores = [d["passage_score"] for d in data_array]   

                # Bar plot for every one of the examples
                for index, result in enumerate(results):
                    if len(result) == 0:
                        continue
                    fig, ax = plt.subplots()
                    ax.bar(range(len(result)), result)

                    # Set ticks from 0 to 1
                    ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

                    ax.set_xticks(range(len(result)))
                    ax.set_xticklabels(range(len(result)))
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(ticks)
                    ax.set_xlabel("Sentence")
                    ax.set_ylabel("SelfCheckGPT Sentence Score")
                    ax.set_title(f"SelfCheckGPT Score for Prompt {int(index)}")
                    plt.savefig(os.path.join(self.result_dir_path, file.split("_")[0], f"prompt_{int(index)}.pdf"), bbox_inches='tight')
                    plt.close()

                # Bar plot for the passage scores
                fig, ax = plt.subplots()
                ax.bar(range(len(passage_scores)), passage_scores)
                ax.set_xticks(range(len(passage_scores)))
                ax.set_xticklabels(range(len(passage_scores)))
                ax.set_xlabel("Prompt")
                ax.set_ylabel("SelfCheckGPT Passage Score")
                ax.set_title("SelfCheckGPT Score for Passages")
                plt.savefig(os.path.join(self.result_dir_path, file.split("_")[0], "passage_scores.pdf"), bbox_inches='tight')
                plt.close()
