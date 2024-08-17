# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List

def read_json_file(filename: str) -> any:
    """
    Read the content of a JSON file.

    :param filename: The name of the file.
    :type filename: str

    :return: The content of the file.
    :rtype: any
    """

    file = open(filename, "r")
    file_content = json.loads(file.read())
    file.close()
    return file_content

def read_runtimes(filename: str, folders: List[str], methods: List[str], embedding_models: List[str]) -> Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]:
    """
    Read the runtimes from the file.
    
    :param filename: The name of the file containing the runtimes.
    :type filename: str
    :param folders: The folders containing the runtimes.
    :type folders: list[str]
    :param methods: The methods used.
    :type methods: list[str]
    :param embedding_models: The embedding models used.
    :type embedding_models: list[str]

    :return: The runtimes.
    :rtype: dict[str, dict[str, dict[str, dict[str, list[float]]]]]
    """

    content = read_json_file(os.path.join(os.path.abspath(os.path.dirname(__file__)), filename))
    data = {}
    for method in methods:
        method_name = {"bert" : "bertscore", "scgpt" : "selfcheckgpt", "ce" : "checkembed"}[method]
        data[method] = {}
        emb_models = embedding_models if method == "ce" else [None]
        for folder in folders:
            data[method][folder] = {}
            for emb_model in emb_models:
                emb_model_name = {"gpt" : "gpt-embedding-large", "sfr" : "sfr-embedding-mistral", "e5" : "e5-mistral-7b-instruct", "gte" : "gte-Qwen15-7B-instruct" , None : ""}[emb_model]
                method_label = method + ("_" + emb_model if emb_model is not None else "")
                data[method][folder][method_label] = {}
                if emb_model_name:
                    data[method][folder][method_label]["labels"] = list(content[folder][method_name][emb_model_name].keys())
                    data[method][folder][method_label]["values"] = [x + y for x , y in zip(list(content[folder][method_name][emb_model_name].values()), list(content[folder]["embedding"][emb_model_name].values()))]
                elif folder != "8_samples" and folder != "10_samples":
                    data[method][folder][method_label]["labels"] = list(content[folder][method_name].keys())
                    data[method][folder][method_label]["values"] = list(content[folder][method_name].values())
                else:
                    data[method][folder][method_label]["labels"] = []
                    data[method][folder][method_label]["values"] = []

    #DEBUG
    # with open("runtimes_results_final.json", "w") as f:
    #     json.dump(data, f, indent=4)

    return data

def plot_performance(filename: str, folders: List[str], methods: List[str], embedding_models: List[str]) -> None:
    """
    Plot the runtimes of the different models and methods varying the number of samples and the lenght of the text.

    :param filename: The name of the file containing the runtimes.
    :type filename: str
    :param folders: The folders containing the runtimes.
    :type folders: list[str]
    :param methods: The methods used.
    :type methods: list[str]
    :param embedding_models: The embedding models used.
    :type embedding_models: list[str]
    """
    method_labels = {"bert" : "BERTScore", "scgpt" : "SelfCheckGPT", "ce" : "CheckEmbed", "ce_gpt" : "CheckEmbed (GPT)", "ce_sfr" : "CheckEmbed (SFR)", "ce_e5" : "CheckEmbed (E5)", "ce_gte" : "CheckEmbed (GTE)"}
    colors = {"bert" : "#999900", "scgpt" : "#990099", "ce_gpt" : "#0000FF", "ce_sfr" : "#FF0000", "ce_e5" : "#00FF00", "ce_gte" : "#9900FF"}

    # Read data
    data = read_runtimes(filename, folders, methods, embedding_models)

    # Create plot
    num_columns = len(folders)
    num_rows = len(methods)
    _, ax = plt.subplots(num_rows, num_columns, figsize=(5.5 * num_columns, 3.5 * num_rows))
    plt.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.1, wspace=0.05, hspace=0.05)

    # Iterate over the data to plot
    for (i, model) in enumerate(list(data.keys())):
        for (j, typ) in enumerate(list(data[model].keys())):
            x = np.arange(len(data[model][typ][list(data[model][typ].keys())[0]]["labels"])) # x-axis positions
            width = 0.2 if model == "ce" else 0.8  # Width of bars
            for (k, method) in enumerate(data[model][typ].keys()):
                # Prepare data
                values = data[model][typ][method]["values"]
                labels = data[model][typ][method]["labels"]
                # Plot
                col = colors[method]
                ax[i][j].bar(x + (k - 1.5) * width if model == "ce" else labels, values, label=method_labels[method], color=col, zorder=3, width=0.8 if model != "ce" else 0.2)

            # Configure subplot
            ax[i][j].set_ylim(0, 1.1 * max(max(max([data[model][typ][method]["values"] for method in data[model][typ].keys()]) for typ in list(data[model].keys()))))
            ax[i][j].set_title(typ.replace("_", " ") if i == 0 else "", fontsize=10)
            ax[i][j].set_xlim(-1, len(data[model][typ][method]["values"]))
            ax[i][j].set_ylabel("Runtime [s]" if j == 0 else "")
            ax[i][j].set_yticklabels(ax[i][j].get_yticklabels() if j == 0 else [])
            ax[i][j].set_xticks([i for i, _ in enumerate(data[model][typ][method]["labels"])])
            ax[i][j].set_xticklabels([label for label in data[model][typ][method]["labels"]] if i == num_rows - 1 else [], fontsize=9, rotation=65, ha = "right")
            ax[i][j].grid(axis='y')
            ax[i][j].set_xlabel("Length of samples [#tokens]" if i == num_rows - 1 else "")
            ax[i][j].legend(loc="upper left", fontsize=10, fancybox=True, shadow=True, ncol=2)

    # Save plot
    plt.savefig("runtime.pdf")


if __name__ == "__main__":
    input_file = "runtimes_results.json"
    folders = ["2_samples", "4_samples", "6_samples", "8_samples", "10_samples"] # To modify if the number of samples changes

    methods = ["bert", "scgpt", "ce"] # To modify if less or more methods are used
    # BERTScore, SelfCheckGPT, CheckEmbed (in order)

    embedding_models = ["gpt", "sfr", "e5", "gte"] # To modify if less or more embedding models are used
    # text-embedding-large, sfr-embedding-mistral, e5-mistral-7b-instruct, gte-Qwen15-7B-instruct (in order)

    plot_performance(input_file, folders, methods, embedding_models)