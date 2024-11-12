# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

# TODO: move with normal plotter

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Union

def read_json_file(filename: str) -> Any:
    """
    Read the content of a JSON file.

    :param filename: The name of the file.
    :type filename: str
    :return: The content of the file.
    :rtype: Any
    """

    file = open(filename, "r")
    file_content = json.loads(file.read())
    file.close()
    return file_content

def read_runtimes(filename: str, folders: List[str], methods: List[str], embedding_models: List[str], scgpt_methods: Union[List[str], None]) -> Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]:
    """
    Read the runtimes from the file.
    
    :param filename: The name of the file containing the runtimes.
    :type filename: str
    :param folders: The folders containing the runtimes.
    :type folders: List[str]
    :param methods: The methods used.
    :type methods: List[str]
    :param embedding_models: The embedding models used.
    :type embedding_models: List[str]
    :param scgpt_methods: The SelfCheckGPT methods used.
    :type scgpt_methods: List[str | None]
    :return: The runtimes.
    :rtype: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]
    """

    content = read_json_file(os.path.join(os.path.abspath(os.path.dirname(__file__)), filename))
    data = {}
    for method in methods:
        method_name = {
            "bert": "bertscore",
            "scgpt": "selfcheckgpt",
            "ce": "checkembed",
        }[method]
        data[method] = {}
        emb_models = embedding_models if method == "ce" else [None] * len(scgpt_methods)
        scgpt_models = scgpt_methods if method == "scgpt" else [None] * len(embedding_models)
        for folder in folders:
            data[method][folder] = {}
            for emb_model, scgpt_model in zip(emb_models, scgpt_models):
                emb_model_name = {
                    "gpt": "gpt-embedding-large",
                    "sfr": "sfr-embedding-mistral",
                    "e5": "e5-mistral-7B-instruct",
                    "gte": "gte-qwen1.5-7B-instruct",
                    "ste400": "stella-en-400M-v5",
                    "ste1.5": "stella-en-1.5B-v5",
                    None: "",
                }[emb_model]
                scgpt_model_name = {
                    "bert": "bertscore",
                    "nli": "nli",
                    None: "",
                }[scgpt_model]
                method_label = method + ("_" + emb_model if emb_model is not None else "_" + scgpt_model if scgpt_model is not None else "")
                data[method][folder][method_label] = {}
                try:
                    if emb_model_name:
                        data[method][folder][method_label]["labels"] = list(content[folder][method_name][emb_model_name].keys())
                        data[method][folder][method_label]["values"] = \
                            [x + y for x , y in zip(list(content[folder][method_name][emb_model_name].values()), 
                                list(content[folder]["embedding"][emb_model_name].values()))]
                    elif scgpt_model_name:
                        data[method][folder][method_label]["labels"] = list(content[folder][method_name + "_" + scgpt_model_name].keys())
                        data[method][folder][method_label]["values"] = list(content[folder][method_name + "_" + scgpt_model_name].values())
                    else:
                        data[method][folder][method_label]["labels"] = list(content[folder][method_name].keys())
                        data[method][folder][method_label]["values"] = list(content[folder][method_name].values())
                except KeyError:
                    data[method][folder][method_label]["labels"] = []
                    data[method][folder][method_label]["values"] = []

    return data

def plot_performance(filename: str, output_filename: str, folders: List[str], methods: List[str], embedding_models: List[str], scgpt_methods: Union[List[str], None]) -> None:
    """
    Plot the runtimes of the different models and methods while varying the number of samples and the length of the text.

    :param filename: The name of the file containing the runtimes.
    :type filename: str
    :param output_filename: The name of the output file.
    :type output_filename: str
    :param folders: The folders containing the runtimes.
    :type folders: List[str]
    :param methods: The methods used.
    :type methods: List[str]
    :param embedding_models: The embedding models used.
    :type embedding_models: list[str]
    :param scgpt_methods: The SelfCheckGPT methods used.
    :type scgpt_methods: list[str | None]
    """
    method_labels = {
        "bert": "BERTScore",
        "scgpt": "SelfCheckGPT",
        "scgpt_bert": "SelfCheckGPT (BERTScore)",
        "scgpt_nli": "SelfCheckGPT (NLI)",
        "ce": "CheckEmbed",
        "ce_gpt": "CheckEmbed (GPT)",
        "ce_sfr": "CheckEmbed (SFR)",
        "ce_e5": "CheckEmbed (E5)",
        "ce_gte": "CheckEmbed (GTE)",
        "ce_ste400": "CheckEmbed (STE400)",
        "ce_ste1.5": "CheckEmbed (STE1.5)",
    }
    colors = {
        "bert": "#999900",
        "scgpt": "#990099",
        "scgpt_bert": "#990099",
        "scgpt_nli": "#009999",
        "ce_gpt": "#0000FF",
        "ce_sfr": "#FF0000",
        "ce_e5": "#00FF00",
        "ce_gte": "#9900FF",
        "ce_ste400": "#FF9900",
        "ce_ste1.5": "#00FFFF",
    }

    # Read data
    data = read_runtimes(filename, folders, methods, embedding_models, scgpt_methods)

    # Create plot
    num_columns = len(folders)
    num_rows = len(methods)
    _, ax = plt.subplots(num_rows, num_columns, figsize=(5.5 * num_columns, 3.5 * num_rows))
    plt.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.1, wspace=0.05, hspace=0.05)

    # Iterate over the data to plot
    for (i, model) in enumerate(list(data.keys())):
        for (j, typ) in enumerate(list(data[model].keys())):
            x = np.arange(len(data[model][typ][list(data[model][typ].keys())[0]]["labels"])) # x-axis positions
            width = 0.8 / len(data[model][typ].keys()) if (model == "ce" or model == "scgpt") else 0.8  # Width of bars
            max_label = max(max(len(data[method][folder][list(data[method][folder].keys())[0]]["labels"]) for method in data.keys()) for folder in data[list(data.keys())[0]].keys())

            for (k, method) in enumerate(data[model][typ].keys()):
                # Prepare data
                values = data[model][typ][method]["values"]
                if len(values) < max_label and len(values) > 0:
                    values.extend([0] * (max_label - len(values)))
                labels = data[model][typ][method]["labels"]
                if len(labels) < max_label and len(labels) > 0:
                    labels.extend([""] * (max_label - len(labels)))
                # Plot
                col = colors[method]
                ax[i][j].bar(x + (k + 0.5 - 0.5 * len(data[model][typ].keys())) * width if model == "ce" or model == "scgpt" else labels, values, label=method_labels[method], color=col, zorder=3, width=width)

            # Configure subplot
            values = []
            for array in [data[model][typ][method]["values"] for method in data[model][typ].keys() for typ in list(data[model].keys())]:
                values.extend(array)
            max_values = max(values)
            ax[i][j].set_ylim(0, 1.1 * max_values)
            ax[i][j].set_title(typ.replace("_", " ") if i == 0 else "", fontsize=10)
            ax[i][j].set_xlim(-1, len(data[model][typ][method]["values"]))
            ax[i][j].set_ylabel("Runtime [s]" if j == 0 else "")
            ax[i][j].set_yticklabels(ax[i][j].get_yticklabels() if j == 0 else [])
            ax[i][j].set_xticks([i for i in range(max_label)])
            ax[i][j].set_xticklabels([label for label in data[model][typ][method]["labels"]] if i == num_rows - 1 else [], fontsize=9, rotation=65, ha = "right")
            ax[i][j].grid(axis='y')
            ax[i][j].set_xlabel("Length of samples [#tokens]" if i == num_rows - 1 else "")
            ax[i][j].legend(loc="upper left", fontsize=10, fancybox=True, shadow=True, ncol=2)

    # Save plot
    plt.savefig(output_filename)


if __name__ == "__main__":
    input_file = "runtimes_results.json"

    output_file = "runtime.pdf"

    # Modify to the used sample numbers
    # The names should correspond to the folders, where the results for a given sample numbers were stored.
    folders = ["2_samples", "4_samples", "6_samples", "8_samples", "10_samples"]

    # Modify to add or remove methods
    # BERTScore, SelfCheckGPT (BERTScore), SelfCheckGPT (NLI), CheckEmbed (in order)
    methods = ["bert", "scgpt", "ce"]

    # Modify to add or remove embedding models
    # text-embedding-large, sfr-embedding-mistral, e5-mistral-7B-instruct, gte-qwen1.5-7B-instruct, stella-en-400M-v5, stella-en-1.5B-v5 (in order)
    embedding_models = ["gpt", "sfr", "e5", "gte", "ste400", "ste1.5"]

    scgpt_models = ["bert", "nli"]

    plot_performance(input_file, output_file, folders, methods, embedding_models, scgpt_models)
