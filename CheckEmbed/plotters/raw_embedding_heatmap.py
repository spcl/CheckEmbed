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

class RawEmbeddingHeatPlot(PlotOperation):
    """
    The RawEmbeddingHeatPlot class handles the plotting of the raw embedding data.

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
        print("Running RawEmbeddingHeatPlot operation.")
        
        for file in os.listdir(self.data_dir_path):
            if ".json" in file and not file.startswith("ground_truth_"):
                
                folder_name = file.replace("_" + file.split("_")[2], "")
                file_name_completion_for_ground_truth = file.replace(file.split("_")[0] + "_", "")

                # Directory creation
                if not os.path.exists(os.path.join(self.result_dir_path, folder_name)):
                    os.mkdir(os.path.join(self.result_dir_path, folder_name))

                if not os.path.exists(os.path.join(self.result_dir_path, folder_name, "raw_embeddings_heat_map")):
                    os.mkdir(os.path.join(self.result_dir_path, folder_name, "raw_embeddings_heat_map"))

                # Load the sample embeddings
                with open(os.path.join(self.data_dir_path, file), "r") as f:
                    data = json.load(f)
                data_array = data["data"]
                embeddings = [d["embeddings"] for d in data_array]  # Convert to numpy array
                # Remove empty ones inside embedding
                for embedding in embeddings:
                    new_embedding = []
                    for index, emb in enumerate(embedding):
                        if len(emb) == 0:
                            continue
                        new_embedding.append(emb)
                    embeddings[embeddings.index(embedding)] = new_embedding
                embeddings = [np.array(embedding) for embedding in embeddings]

                # Load the definition embeddings
                if custom_inputs["ground_truth"]:
                    with open(os.path.join(self.data_dir_path, "ground_truth_" + file_name_completion_for_ground_truth), "r") as f:
                        definitions = json.load(f)
                    definitions = definitions["data"]
                    definitions_embedded = [np.array(d["embeddings"]) for d in definitions]
                    
                    for index, embedding in enumerate(embeddings):
                        if len(embedding) == 0:
                            continue
                        embedding = np.vstack([embedding, definitions_embedded[index].reshape(1, -1)]) if len(definitions_embedded[index]) != 0 else embedding
                        embeddings[index] = embedding

                # Find the min and max values for the colorbar
                min_value = float('inf')
                max_value = float('-inf')

                for embedding in embeddings:
                    if len(embedding) == 0:
                        continue
                    min_value = min(min_value, np.min(embedding))
                    max_value = max(max_value, np.max(embedding))

                # Plot each heatmap
                for index, embedding in enumerate(embeddings):
                    if len(embedding) == 0:
                        continue
                    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the figure size as needed
                    
                    im = ax.imshow(embedding, cmap='YlGnBu', interpolation='nearest', aspect="auto", vmin=min_value, vmax=max_value)
                    plt.colorbar(im, ax=ax)  # Use ax argument to specify the axis for the colorbar

                    plt.title(f"Heatmap of Example {index}", weight='bold', fontsize=26)  # Add a title with index starting from 1
                    plt.xlabel("i-th element of the embedded answers", fontsize=18)
                    plt.ylabel("Embedded Answers", fontsize=18)

                    # Set ticks and labels
                    tick_labels = list(range(1, embedding.shape[0])) + ['GT'] if custom_inputs["ground_truth"] and len(definitions_embedded[index]) > 0 else list(range(1, embedding.shape[0] + 1))
                    ax.set_yticks(np.arange(embedding.shape[0]))
                    ax.set_yticklabels(tick_labels, fontsize=18)
                    
                    plt.savefig(os.path.join(self.result_dir_path, folder_name, "raw_embeddings_heat_map", f"raw_embeddings_heat_map_{index}.pdf"), bbox_inches='tight')
                    plt.close()
