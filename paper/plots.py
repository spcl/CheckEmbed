# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# authors:
# Patrick Iff
# Lorenzo Paleari
# Robert Gerstenberger
# Eric Schreiber


import json
import math
import os
from decimal import Decimal
from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)


# written by Patrick Iff
def read_json_file(filename):
    file = open(filename, "r")
    file_content = json.loads(file.read())
    file.close()
    return file_content


# written by Patrick Iff
def read_file_base(path, method, metric):
    data = read_json_file(path)
    results = []
    if "judge" in path:
        results = [int(value)/100.0 for value in data["data"]]
        return results
    for entry in data["data"]:
        metric_map_score = {"bert" : "frobenius_norm", "scgpt_bert" : "passage_score", "scgpt_nli" : "passage_score", "ce" : "frob_norm_cosine_sim", "ce_got" : "frob_norm_cosine_sim"}
        metric_map_std_dev = {"bert" : "std_dev", "scgpt_bert" : "std_dev", "scgpt_nli" : "std_dev", "ce" : "std_dev_cosine_sim", "ce_got" : "std_dev_cosine_sim"}
        metric_map_map = {"score" : metric_map_score, "stddev" : metric_map_std_dev}
        key = metric_map_map[metric][method]
        if key is not None and key in entry:
            results.append(entry[key])
    return results


# written by Patrick Iff
def read_file(error, model, method, emb_model, judge, metric, file):
    dir1 = file
    dir2 = ("error_%d" % error) if type(error) is int else error
    dir3 = {"bert" : "BertScore", "scgpt_bert" : "SelfCheckGPT", "scgpt_nli": "SelfCheckGPT", "judge": "Judge", "ce" : "CheckEmbed", "ce_got" : "CheckEmbed_self"}[method]
    emb_model2 = {"gpt" : "gpt-embedding-large_results", "sfr" : "sfr-embedding-mistral_results", "e5" : "e5-mistral-7B-instruct_results", "gte" : "gte-qwen1.5-7B-instruct_results", "ste400": "stella-en-400M-v5_results", "ste1.5": "stella-en-1.5B-v5_results",  None : ""}[emb_model]
    judge_model = {"4o": "4o", "4o-mini": "4o-mini", "llama70b": "llama70b", "llama8b": "llama8b"}[judge]
    file = model + "_" + {"bert" : "bert", "scgpt_bert": "selfcheckgpt_BertScore", "scgpt_nli": "selfcheckgpt_NLI", "judge": "judge", "ce" : emb_model2, "ce_got" : emb_model2}[method] + ".json"
    if method == "judge":
        file = judge_model + "_" + model + "_" + method + ".json"
    path = "results/%s/%s/%s/%s" % (dir1, dir2, dir3, file)
    # Hack
    if error == "ground_truth" and method == "ce_got":
        path = path.replace("CheckEmbed_self", "CheckEmbed")
    return read_file_base(path, method, metric)


# written by Patrick Iff
def read_all_files(model, emb_model, judge, metric, file, methods = ["bert", "scgpt_bert", "scgpt_nli", "judge", "ce"]):
    errors = ["ground_truth"] + list(range(1, 11))
    data = {}
    for error in errors:
        data[error] = {}
        for method in methods:
            sub_data =  read_file(error, model, method, emb_model, judge, metric, file)
            if len(sub_data) > 0:
                data[error][method] = sub_data
    return data


# written by Patrick Iff
def do_plot(ax, values, xpos, col):
    tmp = ax.violinplot(values, positions=[xpos], showmeans=True, showmedians=False, widths=1.0)
    # Set colors
    for pc in tmp['bodies']:
        pc.set_facecolor(col)
        pc.set_edgecolor(col)
    for partname in ('cbars','cmins','cmaxes','cmeans'):
        tmp[partname].set_edgecolor(col)
        tmp[partname].set_linewidth(1)


# Read one file containing results
# written by Patrick Iff
def read_description_file(typ, method, model, mode, emb_model):
    dir1 = "description/%s" % typ
    dir1 += "" if typ == "different" else ("/" + mode)
    dir2 = {"bert" : "BertScore", "scgpt" : "SelfCheckGPT", "ce" : "CheckEmbed", "judge": "Judge"}[method]
    emb_model_2 = {"gpt" : "gpt-embedding-large_results", "sfr" : "sfr-embedding-mistral_results", "e5" : "e5-mistral-7B-instruct_results", "gte" : "gte-qwen1.5-7B-instruct_results", "ste400": "stella-en-400M-v5_results", "ste1.5": "stella-en-1.5B-v5_results", "bert": "selfcheckgpt_BertScore", "nli": "selfcheckgpt_NLI", "4o": "4o", "4o-mini": "4o-mini", "llama70b": "llama70b", "llama8b": "llama8b", None: ""}[emb_model]
    file = model + "_" + {"bert" : "bert", "scgpt" : emb_model_2, "ce" : emb_model_2, "judge": "judge"}[method] + ".json"
    if method == "judge":
        file = emb_model_2 + "_" + model + "_" + method + ".json"
    path = "results/%s/%s/%s" % (dir1, dir2, file)
    data = read_json_file(path)
    results = []
    if method == "judge":
        results = [int(value)/100.0 for value in data["data"]]
        return results
    for entry in data["data"]:
        score = entry[{"bert" : "frobenius_norm", "scgpt" : "passage_score", "ce" : "frob_norm_cosine_sim"}[method]]
        results.append(score)
    return results


# Read all files containing results
# written by Patrick Iff
def read_all_description_files(mode):
    types = ["different","similar"]
    methods = ["bert","scgpt","ce","judge"]
    models = ["gpt","gpt4-turbo","gpt4-o"]
    embedding_models = ["gpt","sfr","e5","gte","ste400","ste1.5"]
    scgp_methods = ["bert","nli"]
    judge_models = ["4o","4o-mini","llama70b","llama8b"]
    data = {}
    for model in models:
        data[model] = {}
        for method in methods:
            emb_models = embedding_models if method == "ce" else scgp_methods if method == "scgpt" else judge_models if method == "judge" else [None]
            for emb_model in emb_models:
                method_label = method + ("_" + emb_model if emb_model is not None else "")
                data[model][method_label] = {}
                for typ in types:
                    data[model][method_label][typ] = read_description_file(typ, method, model, mode, emb_model)
    return data


# Create violin plot
# written by Patrick Iff
def plot_description(mode, gpt4o_only = False):
    # Config
    colors = {"different" : "#990000", "similar" : "#009900"}
    method_labels = {
        "bert" : "BERTScore",
        "scgpt" : "SelfCheckGPT",
        "scgpt_bert" : "SelfCheckGPT (BERT)",
        "scgpt_nli" : "SelfCheckGPT (NLI)",
        "judge_4o" : "LLM-as-a-Judge (GPT-4o)",
        "judge_4o-mini" : "LLM-as-a-Judge (GPT-4o-mini)",
        "judge_llama70b" : "LLM-as-a-Judge (LLaMA-70B)",
        "judge_llama8b" : "LLM-as-a-Judge (LLaMA-8B)",
        "ce" : "CheckEmbed",
        "ce_gpt" : "CheckEmbed (GPT)",
        "ce_sfr" : "CheckEmbed (SFR)",
        "ce_e5" : "CheckEmbed (E5)",
        "ce_gte" : "CheckEmbed (GTE)",
        "ce_ste400" : "CheckEmbed (STE400)",
        "ce_ste1.5" : "CheckEmbed (STE1.5)"
        }
    model_labels = {"gpt4-o" : "GPT-4o", "gpt4-turbo" : "GPT-4-turbo", "gpt" : "GPT-3.5"}
    # Read data
    data = read_all_description_files(mode)
    # Create plot
    (fig, ax) = plt.subplots(1, 1 if gpt4o_only else  3, figsize=(4 if gpt4o_only else 12, 5))
    if type(ax) is not np.ndarray:
        ax = [ax]
    fig.subplots_adjust(left=0.18 if gpt4o_only else 0.065, right=0.99, top=0.925, bottom=0.25, wspace=0.05)
    # Iterate through data
    for (i, model) in enumerate((["gpt4-o"] if gpt4o_only else data.keys())):
        legend_patches = set()
        for (j, method) in enumerate(data[model].keys()):
            for (k, typ) in enumerate(data[model][method].keys()):
                # Prepare data
                values = data[model][method][typ]
                xpos = (len(data[model][method].keys()) + 1) * j + k + (-1 if k == 0 else +1) * 0.15
                col = colors[typ]
                # Plot
                tmp = ax[i].violinplot(values, positions=[xpos], showmeans=True, showmedians=False, widths=1.2)
                # Set colors
                for pc in tmp['bodies']:
                    pc.set_facecolor(col)
                    pc.set_edgecolor(col)
                for partname in ('cbars','cmins','cmaxes','cmeans'):
                    tmp[partname].set_edgecolor(col)
                    tmp[partname].set_linewidth(1)
                legend_patches.add((col, typ.capitalize()))
            # Vertical lines between methods
            ax[i].vlines(xpos + 1, 0, 1, color="#000000", lw=0.5)
        # Configure subplot
        ax[i].set_title(model_labels[model], fontsize=10)
        ax[i].set_ylim(0,1)
        ax[i].set_xlim(-1, len(data[model].keys()) * (len(data[model][method].keys()) + 1) - 1)
        ax[i].set_ylabel("Score (higher: assessed as more similar)" if i == 0 else "")
        ax[i].set_yticklabels(ax[i].get_yticklabels() if i == 0 else [])
        width = len(data[model][method].keys()) + 1
        hwidth = width / 2
        ax[i].set_xticks([x * width + hwidth - 1 for x in range(len(data[model].keys()))])
        ax[i].set_xticklabels([method_labels[method] for method in data[model].keys()], fontsize=9, rotation=35, ha = "right")
        ax[i].grid(axis='y')
        # Create legend
        legend_patches = [mpatches.Patch(color=col, label=lab, alpha = 0.5) for (col, lab) in legend_patches]
        ax[i].legend(handles=list(legend_patches), loc='upper left', fontsize=9)
    # Save plot
    #plt.savefig("plot_eval_violins_gpt_%s.png" % mode, format='png', dpi=600)
    plt.savefig("plot_eval_violins_gpt_%s%s.pdf" % (mode , "_gpt4o" if gpt4o_only else ""))


# Read all files containing results
# based on code written originally by Patrick Iff
# adapted by Robert Gerstenberger
def read_all_description_files2():
    modes = ["generic", "precise"]
    types = ["different","similar"]
    methods = ["bert","scgpt","judge","ce"]
    models = ["gpt","gpt4-turbo","gpt4-o"]
    embedding_models = ["gpt","sfr","e5","gte","ste400","ste1.5"]
    scgp_methods = ["bert","nli"]
    judge_models = ["4o","4o-mini","llama70b","llama8b"]
    data = {}
    for mode in modes:
        data[mode] = {}
        for model in models:
            data[mode][model] = {}
            for method in methods:
                emb_models = embedding_models if method == "ce" else scgp_methods if method == "scgpt" else judge_models if method == "judge" else [None]
                for emb_model in emb_models:
                    method_label = method + ("_" + emb_model if emb_model is not None else "")
                    data[mode][model][method_label] = {}
                    for typ in types:
                        data[mode][model][method_label][typ] = read_description_file(typ, method, model, mode, emb_model)
    return data


# Create combined violin plot
# based on code written originally by Patrick Iff
# adapted by Robert Gerstenberger
def plot_description_combined():
    # Config
    colors = {"different" : "#990000", "similar" : "#009900"}
    method_labels = {
        "bert" : "BERTScore",
        "scgpt" : "SelfCheckGPT",
        "scgpt_bert" : "SelfCheckGPT (BERT)",
        "scgpt_nli" : "SelfCheckGPT (NLI)",
        "judge_4o" : "LLM-as-a-Judge (GPT-4o)",
        "judge_4o-mini" : "LLM-as-a-Judge (GPT-4o-mini)",
        "judge_llama70b" : "LLM-as-a-Judge (LLaMA-70B)",
        "judge_llama8b" : "LLM-as-a-Judge (LLaMA-8B)",
        "ce" : "CheckEmbed",
        "ce_gpt" : "CheckEmbed (GPT)",
        "ce_sfr" : "CheckEmbed (SFR)",
        "ce_e5" : "CheckEmbed (E5)",
        "ce_gte" : "CheckEmbed (GTE)",
        "ce_ste400" : "CheckEmbed (STE400)",
        "ce_ste1.5" : "CheckEmbed (STE1.5)"
        }
    model_labels = {"gpt4-o" : "GPT-4o", "gpt4-turbo" : "GPT-4-turbo", "gpt" : "GPT-3.5"}
    # Read data
    data = read_all_description_files2()
    # Create plot
    num_rows = 3
    num_cols = 2
    (fig, ax) = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 5*num_rows))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.925, bottom=0.25, wspace=0.05)

    # Iterate through data
    for (column, mode) in enumerate(data.keys()):
        ax[0, column].set_title(f"{mode}".title(), fontsize=14, weight='bold')
        for (i, model) in enumerate(data[mode].keys()):
            legend_patches = set()
            for (j, method) in enumerate(data[mode][model].keys()):
                for (k, typ) in enumerate(data[mode][model][method].keys()):
                    # Prepare data
                    values = data[mode][model][method][typ]
                    xpos = (len(data[mode][model][method].keys()) + 1) * j + k + (-1 if k == 0 else +1) * 0.15
                    col = colors[typ]
                    # Plot
                    tmp = ax[i, column].violinplot(values, positions=[xpos], showmeans=True, showmedians=False, widths=1.2)
                    # Set colors
                    for pc in tmp['bodies']:
                        pc.set_facecolor(col)
                        pc.set_edgecolor(col)
                    for partname in ('cbars','cmins','cmaxes','cmeans'):
                        tmp[partname].set_edgecolor(col)
                        tmp[partname].set_linewidth(1)
                    legend_patches.add((col, typ.capitalize()))
                # Vertical lines between methods
                ax[i, column].vlines(xpos + 1, 0, 1, color="#000000", lw=0.5)
            # Configure subplot
            # a bit of a hack, so that label is there and can be moved manually
            if column == 0:
                ax2 = ax[i, column].twinx()
                ax2.set_yticks([])
                ax2.set_ylabel(model_labels[model], fontsize=14, weight='bold')
            ax[i, column].set_ylim(0,1)
            ax[i, column].set_xlim(-1, len(data[mode][model].keys()) * (len(data[mode][model][method].keys()) + 1) - 1)
            if column > 0:
                # Remove ticks on y axis apart from the first column, but keep the grid lines
                for tick in ax[i, column].yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
            ax[i, column].set_ylabel("Score (higher: assessed as more similar)" if column == 0 else "")
            ax[i, column].set_yticklabels(ax[i, column].get_yticklabels() if column == 0 else [])
            width = len(data[mode][model][method].keys()) + 1
            hwidth = width / 2
            ax[i, column].set_xticks([x * width + hwidth - 1 for x in range(len(data[mode][model].keys()))] if i == num_rows-1 else [])
            ax[i, column].set_xticklabels([method_labels[method] for method in data[mode][model].keys()] if i== num_rows-1 else [], fontsize=9, rotation=35, ha = "right")
            ax[i, column].grid(axis='y')
            # Create legend
            legend_patches = [mpatches.Patch(color=col, label=lab, alpha = 0.5) for (col, lab) in legend_patches]
            ax[i, column].legend(handles=list(legend_patches), loc='upper left', fontsize=9)
    # Save plot
    plt.savefig("plot_eval_violins_combined.pdf")


# written by Patrick Iff
def plot_hallucination(model, emb_model, judge, metric):
    # Config
    colors = {"bert" : "#999900", "scgpt_bert" : "#990099", "scgpt_nli": "#009999" ,"ce" : "#000099", "ce_got" : "#000099", "judge": "#990000"}
    method_labels = {"GOT" : "GOT", "bert" : "BERTScore", "scgpt_bert" : "SelfCheckGPT (BERT)", "scgpt_nli": "SelfCheckGPT (NLI)", "ce" : "CheckEmbed (STE1.5)", "ce_got" : "CheckEmbed", "judge": "LLM-as-a-Judge (LLaMA-8B)"}
    # Read data
    data = read_all_files(model, emb_model, judge, metric, "incremental_forced_hallucination/scientific_descriptions")
    # Create plot
    (fig, ax) = plt.subplots(1, 1, figsize=(9, 3.25))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.135)
    legend_patches = set()
    # Iterate through data
    for (i, error) in enumerate(data.keys()):
        for (j, method) in enumerate(data[error].keys()):
            # Prepare data
            values = data[error][method]
            xpos = i * (len(data[error].keys()) + 1) + j
            col = colors[method]
            do_plot(ax, values, xpos, col)
            legend_patches.add((col, method_labels[method]))
        # Vertical lines between methods
        ax.vlines(xpos + 1, 0, 1, color="#000000", lw=0.5)
    # Configure plot
    ax.set_xlim(-1,xpos+1)
    ax.grid(axis = "y")
    ax.set_xlabel(" " * 17 + "Number of introduced Errors")
    if metric == "score":
        ax.set_ylim(0,1)
        ax.set_ylabel("Score (higher: assessed as more similar)")
    elif metric == "stddev":
        ax.set_ylim(0,math.ceil(10 * max([max([max(data[error][method]) for method in data[error].keys()]) for error in data.keys()])) / 10)
        ax.set_ylabel("Standard deviation (lower: more consistent)")
    else:
        print("Unknown metric: %s" % metric)
    # Set x-ticks
    ax.set_xticks([(x+0.5) * (len(data[1]) + 1) - 1 for x in range(len(data.keys()))])
    ax.set_xticklabels([str(x).replace("got_data","Ground Truth") for x in data.keys()])
    # Create legend
    legend_patches = [mpatches.Patch(color=col, label=lab, alpha = 0.5) for (col, lab) in legend_patches]
    legend_patches = sorted(legend_patches, key=lambda x: list(method_labels.values()).index(x.get_label()))
    ax.legend(handles=list(legend_patches), loc='lower center', fontsize=9, ncol = 5, bbox_to_anchor=(0.5, 0.985))

    #plt.savefig("plot_halucinate_%s_%s_%s.png" % (model, emb_model, metric), format='png', dpi=600)
    plt.savefig("plot_halucinate_%s_%s_%s.pdf" % (model, emb_model, metric))


# written by Robert Gerstenberger
# with contributions by Lorenzo Paleari
def plot_heatmap(filename_checkembed, filename_bertscore, selection, output_filename):
    fig_fontsize = 18

    with open(filename_checkembed, "r") as f:
        data = json.load(f)
    data_array = data["data"]
    cosine_similarity_matrix_array_checkembed = [np.array(d["cosine_sim"]) for d in data_array]  # Convert to numpy array

    with open(filename_bertscore, "r") as f:
        data = json.load(f)
    data_array = data["data"]
    cosine_similarity_matrix_array_bertscore = [np.array(d["result"]) for d in data_array]  # Convert to numpy array

    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    fig.set_figheight(20)
    fig.set_figwidth(20)

    axs[0, 0].set_title("CheckEmbed", fontsize=fig_fontsize+8, weight='bold', color='#008000')
    axs[0, 1].set_title("BERTScore", fontsize=fig_fontsize+8, weight='bold', color='#800000')

    row_labels = ["High Confidence", "Low Confidence"]

    for row in range(2):
        index = selection[row]
        for col in range(2):
            if col == 0:
                cosine_similarity_matrix = cosine_similarity_matrix_array_checkembed[index]
            else:
                cosine_similarity_matrix = cosine_similarity_matrix_array_bertscore[index]
                ax2 = axs[row, col].twinx()
                ax2.set_yticks([])
                ax2.set_ylabel(f"Legal Example {index} ({row_labels[row]})", fontsize=fig_fontsize+8, weight='bold')

            im = axs[row, col].imshow(cosine_similarity_matrix, cmap='YlGnBu', interpolation='nearest', aspect="auto", vmin=-1, vmax=1)

            # Set ticks and labels
            tick_labels = list(range(1, len(cosine_similarity_matrix))) + ['GT']
            if row == 1:
                axs[row, col].set_xticks(np.arange(len(cosine_similarity_matrix)))
                axs[row, col].set_xticklabels(tick_labels, fontsize=fig_fontsize)
                axs[row, col].set_xlabel("LLM Reply ID or Ground-Truth (GT)", fontsize=fig_fontsize)
            else:
                axs[row, col].set_xticks([])
            if col == 0:
                axs[row, col].set_yticks(np.arange(len(cosine_similarity_matrix)))
                axs[row, col].set_yticklabels(tick_labels, fontsize=fig_fontsize)
                axs[row, col].set_ylabel("LLM Reply ID or Ground-Truth (GT)", fontsize=fig_fontsize)
            else:
                axs[row, col].set_yticks([])

            # Add numbers to the heatmap
            for i in range(len(cosine_similarity_matrix)):
                for j in range(len(cosine_similarity_matrix)):
                    axs[row, col].text(j, i, round(Decimal(cosine_similarity_matrix[i, j]), 2), ha="center", va="center", color="white", fontsize=fig_fontsize, weight='bold')

    cbar = plt.colorbar(im, ax=axs, aspect=80, orientation = 'horizontal')
    cbar.ax.tick_params(labelsize=fig_fontsize)

    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()


# written by Lorenzo Paleari
def read_runtimes(filename: str, folders: List[str], methods: List[str], embedding_models: List[str], scgpt_methods: List[Optional[str]]) -> Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]:
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
    :type scgpt_methods: List[Optional[str]]

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


# written by Patrick Iff
def plot_runtime(filename_in, filename_out, folders, methods, ce_models, scgpt_models, is_old = False):
    # Plot config
    method_info = {
        # Used in the main plot
        "bert": ("BERTScore", "#999900", "o"),
        "scgpt_bert": ("SelfCheckGPT (BERT)", "#990099", "s"),
        "scgpt_nli": ("SelfCheckGPT (NLI)", "#009999", "D"),
        "ce_ste400": ("CheckEmbed (STE400)", "#009900", "v"),
        "ce_ste1.5": ("CheckEmbed (STE1.5)", "#000099", "^"),
        # Used in the old plot for the appendix
        "ce_gpt": ("CheckEmbed (GPT)", "#000099", "D"),
        "ce_sfr": ("CheckEmbed (SFR)", "#009900", "v"),
        "ce_e5": ("CheckEmbed (E5)", "#000000", "^"),
        "ce_gte": ("CheckEmbed (GTE)", "#990000", "*"),
    }
    # Dictionary with the list of models for a given method and with the corresponding keys
    models_per_method = {"bert" : [None], "scgpt" : scgpt_models, "ce" : ce_models}
    is_odd = len(folders) % 2 == 1

    # Read data
    data = read_runtimes(filename_in, folders, methods, ce_models, scgpt_models)

    # Initialize Plot
    fig, ax = plt.subplots(1, len(folders), figsize=(4 * len(folders), 3.5))
    fig.subplots_adjust(left=0.05 if is_odd else 0.1, right=0.99, top=0.85, bottom=0.2, wspace=0.125)

    # Iterate over sample counts (subplots)
    for (i, sample_count) in enumerate(folders):
        # Iterate over methods and models
        # Each method-model-combination is a different data series
        for (j1, method) in enumerate(methods):
            models = models_per_method[method]
            for (j2, model) in enumerate(models):
                key = (method + "_" + model) if model else method
                info_key = "scgpt_bert" if key == "scgpt" else key
                # Read the data
                token_counts = [int(x) for x in data[method][sample_count][key]["labels"]]
                runtimes = data[method][sample_count][key]["values"]
                (lab, col, mar) = method_info[info_key]
                ax[i].plot(token_counts, runtimes, label=lab, color=col, marker = mar, markersize = 5)
        # Configure the plot (general)
        ax[i].grid()
        ax[i].set_title(sample_count.split("_")[0] + " Samples", fontsize=10)
        # Configure x-axis
        ax[i].set_xticks(list(range(200,4200,200)))
        ax[i].set_xticklabels(list(range(200,4200,200)), rotation=90)
        ax[i].set_xlabel("Number of tokens")
        ax[i].set_xlim(200, 4000)
        # Configure y-axis
        ax[i].set_yscale("log")
        if i > 0:
            ax[i].set_yticks(ax[i].get_yticks())
            ax[i].set_yticklabels(["" for _ in ax[i].get_yticks()])
        ax[i].set_ylim(10 if is_old else 1,(10000 if is_odd else 1000) if is_old else 20000)
        ax[i].set_ylabel("Runtime [s]" if i == 0 else "")
    # Legend
    sp = int((len(folders)-1) / 2)
    xpos = 0.45 if is_odd else 0.95
    ax[sp].legend(loc="upper center", bbox_to_anchor=(xpos, 1.23), ncol=6 if is_odd else 4, fontsize=10, frameon=False)
    # Save plot
    plt.savefig(filename_out)


# written by Lorenzo Paleari
def load_data(curr_dir: str, sample_accuracy: bool = False):
    # Read data
    passage_scores = None
    with open(os.path.join(curr_dir, "data", "passage_scores.json"), "r") as f:
        passage_scores = [value for value in json.load(f).values()]

    scgpt_ce = None
    scgpt_nli = None
    judge_scores = {}
    judge_scores_ref = {}
    with open(os.path.join(curr_dir, "20_samples", "SelfCheckGPT", "wikibio_selfcheckgpt_BertScore.json"), "r") as f:
        scgpt_ce = [value["passage_score"] for value in json.load(f)["data"]]
    with open(os.path.join(curr_dir, "20_samples", "SelfCheckGPT", "wikibio_selfcheckgpt_NLI.json"), "r") as f:
        scgpt_nli = [value["passage_score"] for value in json.load(f)["data"]]
    for m in ["4o", "4o-mini", "llama70b", "llama8b"]:
        with open(os.path.join(curr_dir, "Judge", f"{m}_judge.json"), "r") as f:
            judge_scores[m] = json.load(f)["data"]
            judge_scores[m] = [int(value)/100.0 for value in judge_scores[m]]
        with open(os.path.join(curr_dir, "Judge", f"{m}_judge_ref.json"), "r") as f:
            judge_scores_ref[m] = json.load(f)["data"]
            judge_scores_ref[m] = [int(value)/100.0 for value in judge_scores_ref[m]]

    check_embed_scores = {}
    for i in range(2, 22, 2):
        sample_scores = {}
        for file in os.listdir(os.path.join(curr_dir, f"{i}_samples", "CheckEmbed")):
            name = file.split("_")[1]
            with open(os.path.join(curr_dir, f"{i}_samples", "CheckEmbed", file), "r") as f:
                sample_scores.update({
                    name: [value["frob_norm_cosine_sim"] for value in json.load(f)["data"]]
                })

        check_embed_scores.update({
            f"{i}_samples": sample_scores
        })

    bert_score = None
    with open(os.path.join(curr_dir, "20_samples", "BertScore", "wikibio_bert.json"), "r") as f:
        bert_score = [value["frobenius_norm"] for value in json.load(f)["data"]]
    
    if sample_accuracy:
        return passage_scores, scgpt_ce, scgpt_nli, check_embed_scores, bert_score
    return passage_scores, scgpt_ce, scgpt_nli, check_embed_scores, bert_score, judge_scores, judge_scores_ref


# written by Lorenzo Paleari
def compute_final_value(array1, array2):
    score_p, _ = pearsonr(array1, array2)
    score_s, _ = spearmanr(array1, array2)
    return score_p * 100, score_s * 100


# written by Lorenzo Paleari
def plot_samples_accuracy(dir: str, output_name: str):
    method_info = {
        "stella-en-400M-v5": ("STE400", "#009900", "v"),
        "stella-en-1.5B-v5": ("STE1.5", "#000099", "^"),
        "gpt-embedding-large": ("GPT", "#990000", "*"),
        "sfr-embedding-mistral": ("SFR", "#000000", "p"),
        "e5-mistral-7B-instruct": ("E5", "#666666", "h"),
        "gte-qwen1.5-7B-instruct": ("GTE", "#996600", "d"),
    }

    passage_scores, _, _, check_embed_scores, _ = load_data(dir, sample_accuracy=True)
    embedding_methods = list(next(iter(check_embed_scores.values())).keys())

    accuracy = {}
    for sample_scores in check_embed_scores.values():
        for embedding in embedding_methods:
            # Compute correlation for each embedding
            _, spearman_corr = compute_final_value(passage_scores, sample_scores[embedding])
            if embedding not in accuracy:
                accuracy[embedding] = []
            accuracy[embedding].append(spearman_corr)

    fig, ax = plt.subplots(figsize=(4, 3))
    fig.subplots_adjust(left=0.12, right=0.99, top=0.975, bottom=0.15)

    for embedding, values in accuracy.items():
        ax.plot(list(range(2, 22, 2)), values, label=method_info[embedding][0], color=method_info[embedding][1], marker=method_info[embedding][2], markersize=4, linestyle='--')

    ax.set_xlim(1, 21)
    ax.set_xticks(list(range(2,22,2)))
    ax.set_xticklabels(list(range(2,22,2)))
    ax.set_yticks(list(range(60, 80, 2)))
    ax.set_yticklabels(list(range(60, 80, 2)))
    ax.set_xlabel("Number of samples")
    ax.set_ylim(60, 78)
    ax.set_ylabel("Spearman's rank correlation")
    ax.grid(axis='y')

    ax.legend(loc="lower right", bbox_to_anchor=(1, 0), ncol=1, fontsize=10)
    plt.savefig(output_name)


# written by Lorenzo Paleari
def plot_wiki_bio(curr_dir: str, output_name: str):
    passage_scores, scgpt_ce, scgpt_nli, check_embed_scores, bert_score, judge_scores, judge_scores_ref = load_data(curr_dir)

    # Create Markdown table
    # embedding_methods = list(next(iter(check_embed_scores.values())).keys())
    # Provide a hard-coded list to match the paper table
    embedding_methods = ['sfr-embedding-mistral', 'stella-en-400M-v5', 'stella-en-1.5B-v5', 'gpt-embedding-large', 'e5-mistral-7B-instruct', 'gte-qwen1.5-7B-instruct']

    # Header row: Embeddings (5 names) each with Pearson and Spearman columns
    header = "| Samples | Pearson | Spearman |\n"
    ce_header = "| Samples |"
    for embedding in embedding_methods:
        ce_header += f" {embedding} (Pearson) | {embedding} (Spearman) |"
    ce_header += "\n"

    # Alignments for Markdown table
    alignment = "| :---: | :---: | :---: |\n"
    ce_alignment = "| :---: |" + " :---: | :---: |" * len(embedding_methods) + "\n"

    # Create table data, starting with SCGPT (20 samples)
    rows = []
    ce_rows = []

    # SCGPT 20-sample row
    scgpt_p_ce, scgpt_s_ce = compute_final_value(passage_scores, scgpt_ce)
    scgpt_p_nli, scgpt_s_nli = compute_final_value(passage_scores, scgpt_nli)
    bert_p, bert_s = compute_final_value(passage_scores, bert_score)
    judge_p, judge_s = {}, {}
    judge_ref_p, judge_ref_s = {}, {}
    for m in ["4o", "4o-mini", "llama70b", "llama8b"]:
        judge_p[m], judge_s[m] = compute_final_value(passage_scores, judge_scores[m])
        judge_ref_p[m], judge_ref_s[m] = compute_final_value(passage_scores, judge_scores_ref[m])

    # Add SCGPT row
    scgpt_row_ce = f"| {scgpt_p_ce:.4f} | {scgpt_s_ce:.4f} |"
    scgpt_row_nli = f"| {scgpt_p_nli:.4f} | {scgpt_s_nli:.4f} |"
    bert_row = f"| {bert_p:.4f} | {bert_s:.4f} |"
    judge_row = {}
    judge_ref_row = {}
    for m in ["4o", "4o-mini", "llama70b", "llama8b"]:
        judge_row[m] = f"| {judge_p[m]:.4f} | {judge_s[m]:.4f} |"
        judge_ref_row[m] = f"| {judge_ref_p[m]:.4f} | {judge_ref_s[m]:.4f} |"
    

    rows.append("| SCGPT_BertScore " + scgpt_row_ce)
    rows.append("| SCGPT_NLI " + scgpt_row_nli)
    rows.append("| BertScore " + bert_row)
    for m in ["4o", "4o-mini", "llama70b", "llama8b"]:
        rows.append(f"| LLM-as-a-Judge ({m}) " + judge_row[m])
        rows.append(f"| LLM-as-a-Judge w/reference ({m}) " + judge_ref_row[m])

    # Add rows for CheckEmbed scores for 2 to 20 samples
    for sample_key, sample_scores in check_embed_scores.items():
        ce_row = f"| {sample_key} |"
        for embedding in embedding_methods:
            # Compute correlation for each embedding
            pearson_corr, spearman_corr = compute_final_value(passage_scores, sample_scores[embedding])
            ce_row += f" {pearson_corr:.4f} | {spearman_corr:.4f} |"

        ce_rows.append(ce_row)

    # Join all parts into the full markdown table
    ce_markdown_table = ce_header + ce_alignment + "\n".join(ce_rows)
    markdown_table = header + alignment + "\n".join(rows)

    with open(output_name, "w") as f:
        f.write(markdown_table + "\n\n\n" + ce_markdown_table)
        
# written by Lorenzo Paleari
def load_rag_data(curr_dir: str):
    hallu_scores = {}
    hallu_scores["total"] = {}
    scgpt_scores = {}
    scgpt_scores["total"] = {}
    judge_scores = {}
    judge_scores["total"] = {}
    ce_scores = {}
    ce_scores["total"] = {}
    bert_scores = {}
    bert_scores["total"] = []
    correct_totals = []
    for task in ["qa", "summary", "data2text"]:
        hallu_scores[task] = {}
        scgpt_scores[task] = {}
        judge_scores[task] = {}
        ce_scores[task] = {}
        bert_scores[task] = []

        with open(os.path.join(curr_dir, "data", f"{task}_response.json"), "r") as f:
            correct_data = json.load(f)["data"]

        correct_data_bin = [0.0 if len(d["labels"]) > 0 else 1.0 for d in correct_data]
        correct_totals.extend(correct_data_bin)

        # Hallu Detection
        for file in os.listdir(os.path.join(curr_dir, "HalluDetect", f"{task}")):
            name = file.split("mtp")[0]
            with open(os.path.join(curr_dir, "HalluDetect", f"{task}", file), "r") as f:
                data = json.load(f)["output"]

            data = [value[0] for value in data]

            if name not in hallu_scores["total"]:
                hallu_scores["total"][name] = []
            hallu_scores["total"][name].extend(data)
            hallu_scores[task][name] = [precision_score(correct_data_bin, data), recall_score(correct_data_bin, data), f1_score(correct_data_bin, data)]

        # SelfCheckGPT
        for file in os.listdir(os.path.join(curr_dir, "SelfCheckGPT")):
            if file.split("_")[0] != task:
                continue
            name = file.split("_")[2]
            with open(os.path.join(curr_dir, "SelfCheckGPT", file), "r") as f:
                data = json.load(f)["data"]

            data = [value["passage_score"] for value in data]
            data_bin = [1.0 if value > 0.5 else 0.0 for value in data]
            if name not in scgpt_scores["total"]:
                scgpt_scores["total"][name] = []
            scgpt_scores["total"][name].extend(data_bin)
            scgpt_scores[task][name] = [precision_score(correct_data_bin, data_bin), recall_score(correct_data_bin, data_bin), f1_score(correct_data_bin, data_bin)]

        # Judge
        for file in os.listdir(os.path.join(curr_dir, "Judge")):
            if file.split("_")[1] != task:
                continue
            name = file.split("_")[0]
            with open(os.path.join(curr_dir, "Judge", file), "r") as f:
                data = json.load(f)["data"]

            data = [float(value)/100.0 for value in data]
            data_bin = [1.0 if value > 0.5 else 0.0 for value in data]
            if name not in judge_scores["total"]:
                judge_scores["total"][name] = []
            judge_scores["total"][name].extend(data_bin)
            judge_scores[task][name] = [precision_score(correct_data_bin, data_bin), recall_score(correct_data_bin, data_bin), f1_score(correct_data_bin, data_bin)]

        for file in os.listdir(os.path.join(curr_dir, "CheckEmbed")):
            if file.split("_")[0] != task:
                continue
            name = file.split("_")[1]
            with open(os.path.join(curr_dir, "CheckEmbed", file), "r") as f:
                data = json.load(f)["data"]

            data = [value["frob_norm_cosine_sim"] for value in data]
            data_bin = [1.0 if value > 0.5 else 0.0 for value in data]
            if name not in ce_scores["total"]:
                ce_scores["total"][name] = []
            ce_scores["total"][name].extend(data_bin)
            ce_scores[task][name] = [precision_score(correct_data_bin, data_bin), recall_score(correct_data_bin, data_bin), f1_score(correct_data_bin, data_bin)]

        for file in os.listdir(os.path.join(curr_dir, "BertScore")):
            if file.split("_")[0] != task:
                continue
            with open(os.path.join(curr_dir, "BertScore", file), "r") as f:
                data = json.load(f)["data"]

            data = [value["frobenius_norm"] for value in data]
            data_bin = [1.0 if value > 0.5 else 0.0 for value in data]
            bert_scores["total"].extend(data_bin)
            bert_scores[task] = [precision_score(correct_data_bin, data_bin), recall_score(correct_data_bin, data_bin), f1_score(correct_data_bin, data_bin)]

    # Compute the total scores
    for model in hallu_scores["total"].keys():
        hallu_scores["total"][model] = [precision_score(correct_totals, hallu_scores["total"][model]), recall_score(correct_totals, hallu_scores["total"][model]), f1_score(correct_totals, hallu_scores["total"][model])]
    for model in scgpt_scores["total"].keys():
        scgpt_scores["total"][model] = [precision_score(correct_totals, scgpt_scores["total"][model]), recall_score(correct_totals, scgpt_scores["total"][model]), f1_score(correct_totals, scgpt_scores["total"][model])]
    for model in judge_scores["total"].keys():
        judge_scores["total"][model] = [precision_score(correct_totals, judge_scores["total"][model]), recall_score(correct_totals, judge_scores["total"][model]), f1_score(correct_totals, judge_scores["total"][model])]
    for model in ce_scores["total"].keys():
        ce_scores["total"][model] = [precision_score(correct_totals, ce_scores["total"][model]), recall_score(correct_totals, ce_scores["total"][model]), f1_score(correct_totals, ce_scores["total"][model])]
    bert_scores["total"] = [precision_score(correct_totals, bert_scores["total"]), recall_score(correct_totals, bert_scores["total"]), f1_score(correct_totals, bert_scores["total"])]

    return hallu_scores, scgpt_scores, judge_scores, ce_scores, bert_scores

# written by Lorenzo Paleari
def plot_RAGTruth(curr_dir: str, output_name: str):
    hallu_scores, scgpt_scores, judge_scores, ce_scores, bert_scores = load_rag_data(curr_dir)
    # Create Markdown table

    # Header row: Summary QA Data2txt for each one Precision Recall F1.
    # Fist row only model and the 3 tasks
    # secod row has for each task the 3 type of scores
    header = "| Model |  | Summary |  | |  | QA | | | | Data2txt | | | | Total | |\n"
    header +=  "| :---: |" + " :---: |" * 15 + "\n"
    header += "| | Precision | Recall | F1 | | Precision | Recall | F1 | | Precision | Recall | F1 | | Precision | Recall | F1 |\n"
    
    # Fill all the results now
    rows = []
    rows.append("| HalluDetection |  |  |  |  |  |  |  | | | |  |  |  |  |  |\n")
    for model in hallu_scores["summary"].keys():
        rows.append(f"| {model} | {hallu_scores['summary'][model][0]:.4f} | {hallu_scores['summary'][model][1]:.4f} | {hallu_scores['summary'][model][2]:.4f} || {hallu_scores['qa'][model][0]:.4f} | {hallu_scores['qa'][model][1]:.4f} | {hallu_scores['qa'][model][2]:.4f} || {hallu_scores['data2text'][model][0]:.4f} | {hallu_scores['data2text'][model][1]:.4f} | {hallu_scores['data2text'][model][2]:.4f} || {hallu_scores['total'][model][0]:.4f} | {hallu_scores['total'][model][1]:.4f} | {hallu_scores['total'][model][2]:.4f} |\n")
    rows.append("|  |  |  |  |  |  |  |  |  |  | | | | |  |  |\n")
    rows.append("| SelfCheckGPT |  |  |  |  |  |  |  |  | | ||  |  |  |  |\n")
    for model in scgpt_scores["summary"].keys():
        rows.append(f"| {model} | {scgpt_scores['summary'][model][0]:.4f} | {scgpt_scores['summary'][model][1]:.4f} | {scgpt_scores['summary'][model][2]:.4f} || {scgpt_scores['qa'][model][0]:.4f} | {scgpt_scores['qa'][model][1]:.4f} | {scgpt_scores['qa'][model][2]:.4f} || {scgpt_scores['data2text'][model][0]:.4f} | {scgpt_scores['data2text'][model][1]:.4f} | {scgpt_scores['data2text'][model][2]:.4f} || {scgpt_scores['total'][model][0]:.4f} | {scgpt_scores['total'][model][1]:.4f} | {scgpt_scores['total'][model][2]:.4f} |\n")
    rows.append("|  |  |  |  |  |  |  |  |  |  ||||  |  |  |\n")
    rows.append("| LLM-as-a-Judge |  |  |  |  |  | ||| |  |  |  |  |  |  |\n")
    for model in judge_scores["summary"].keys():
        rows.append(f"| {model} | {judge_scores['summary'][model][0]:.4f} | {judge_scores['summary'][model][1]:.4f} | {judge_scores['summary'][model][2]:.4f} || {judge_scores['qa'][model][0]:.4f} | {judge_scores['qa'][model][1]:.4f} | {judge_scores['qa'][model][2]:.4f}| | {judge_scores['data2text'][model][0]:.4f} | {judge_scores['data2text'][model][1]:.4f} | {judge_scores['data2text'][model][2]:.4f} || {judge_scores['total'][model][0]:.4f} | {judge_scores['total'][model][1]:.4f} | {judge_scores['total'][model][2]:.4f} |\n")
    rows.append("|  |  |  |  |  |  |  |  |  |  | ||| |  |  |\n")
    rows.append("| CheckEmbed |  |  |  |  |  |  | ||| |  |  |  |  |  |\n")
    for model in ce_scores["summary"].keys():
        rows.append(f"| {model} | {ce_scores['summary'][model][0]:.4f} | {ce_scores['summary'][model][1]:.4f} | {ce_scores['summary'][model][2]:.4f} | |{ce_scores['qa'][model][0]:.4f} | {ce_scores['qa'][model][1]:.4f} | {ce_scores['qa'][model][2]:.4f} || {ce_scores['data2text'][model][0]:.4f} | {ce_scores['data2text'][model][1]:.4f} | {ce_scores['data2text'][model][2]:.4f} || {ce_scores['total'][model][0]:.4f} | {ce_scores['total'][model][1]:.4f} | {ce_scores['total'][model][2]:.4f} |\n")
    rows.append("|  |  |  |  |  |  |  |  |  |  | ||| |  |  |\n")
    rows.append("| BertScore |  |  |  |  |  |  | ||| |  |  |  |  |  |\n")
    rows.append(f"| BertScore | {bert_scores['summary'][0]:.4f} | {bert_scores['summary'][1]:.4f} | {bert_scores['summary'][2]:.4f} || {bert_scores['qa'][0]:.4f} | {bert_scores['qa'][1]:.4f} | {bert_scores['qa'][2]:.4f} || {bert_scores['data2text'][0]:.4f} | {bert_scores['data2text'][1]:.4f} | {bert_scores['data2text'][2]:.4f} || {bert_scores['total'][0]:.4f} | {bert_scores['total'][1]:.4f} | {bert_scores['total'][2]:.4f} |\n")

    # Join all parts into the full markdown table
    markdown_table = header +  "".join(rows)
    with open(output_name, "w") as f:
        f.write(markdown_table)

# written by Eric Schreiber
def load_all_single_values(file_path):
    """
    Load all single values from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing the single values.
    
    Returns:
            "frob_norm_cosine_sim": float,
            "std_dev_cosine_sim": float,
            "frob_norm_pearson_corr": float,
            "std_dev_pearson_corr": float,
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r') as f:
        file = json.load(f)
        frob_norm_cosine_sim = file["data"][0]["frob_norm_cosine_sim"]
        std_dev_cosine_sim = file["data"][0]["std_dev_cosine_sim"]
        frob_norm_pearson_corr = file["data"][0]["frob_norm_pearson_corr"]
        std_dev_pearson_corr = file["data"][0]["std_dev_pearson_corr"]
    
    return frob_norm_cosine_sim, std_dev_cosine_sim, frob_norm_pearson_corr, std_dev_pearson_corr

def plot_vision_results(correct_images, path):
    frobnorms = []
    std_dev_cosine_sims = []
    frob_norm_pearson_corrs = []
    std_dev_pearson_corrs = []
    for i in range(8):
        for j in range(5):
            file = f"countingitems{i}-{j}.json_results.json"
            if not os.path.exists(os.path.join(path, file)):
                continue
            
            frobnorm, std_dev_cosine_sim, frob_norm_pearson_corr, std_dev_pearson_corr = load_all_single_values(os.path.join(path, file))
            if j == 0:
                frobnorms.append([frobnorm])
                std_dev_cosine_sims.append([std_dev_cosine_sim])
                frob_norm_pearson_corrs.append([frob_norm_pearson_corr])
                std_dev_pearson_corrs.append([std_dev_pearson_corr])
            else:
                frobnorms[-1].append(frobnorm)
                std_dev_cosine_sims[-1].append(std_dev_cosine_sim)
                frob_norm_pearson_corrs[-1].append(frob_norm_pearson_corr)
                std_dev_pearson_corrs[-1].append(std_dev_pearson_corr)


    # Normalize the values for easier comparison as the values are on different scales.
    # As the plot does not allow to show two axes, we normalize the values.

    frobnorms_np0_1 = np.array(frobnorms)
    correct_images_np0_1 = np.array(correct_images)
    # Normalize the values to be between 0 and 1
    frobnorms_normalized = (frobnorms_np0_1 - np.mean(frobnorms_np0_1, axis=1, keepdims=True)) / np.std(frobnorms_np0_1, axis=1, keepdims=True)
    correct_images_normalized = (correct_images_np0_1 - np.mean(correct_images_np0_1, axis=1, keepdims=True)) / np.std(correct_images_np0_1, axis=1, keepdims=True)

    # Create a new DataFrame for the normalized values
    frobnorms_df = pd.DataFrame(frobnorms_normalized, columns=["1 Item", "2 Items", "3 Items", "4 Items", "5 Items"])
    correct_images_df = pd.DataFrame(correct_images_normalized, columns=["1 Item", "2 Items", "3 Items", "4 Items", "5 Items"])
    # Melt the DataFrames to long format
    frobnorms_df = pd.melt(frobnorms_df)
    correct_images_df = pd.melt(correct_images_df)
    # Rename columns for clarity
    frobnorms_df.columns = ["level", "frobnorm"]
    correct_images_df.columns = ["level", "correct_images"]

    # Rename value columns to match for unification
    frobnorms_df_renamed = frobnorms_df.rename(columns={"frobnorm": "value"})
    correct_images_df_renamed = correct_images_df.rename(columns={"correct_images": "value"})

    # Add a column to distinguish types
    frobnorms_df_renamed["type"] = "CheckEmbed Score"
    correct_images_df_renamed["type"] = "Num Correct Images"

    # Combine the dataframes
    combined_df = pd.concat([frobnorms_df_renamed, correct_images_df_renamed], ignore_index=True)

    colors = [(0, 0, 142), (142, 142, 3)]
    colors = [tuple(c/255 for c in color) for color in colors]
    # Plot with Seaborn
    plt.figure(figsize=(8,5))
    sns.violinplot(data=combined_df, x="level", y="value", hue="type", inner="quartile", palette=colors,
                    alpha=0.4, split=True, gap=0.0)
    # make fontsize 14
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.ylabel("Score", fontsize=15)
    plt.xlabel("Number of Items in the Image", fontsize=15)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig("./frobnorms_correct_images_normalized.pdf")

plot_description_combined()

plot_description(
    "generic"
)

plot_description(
    "precise"
)

plot_heatmap(
    "results/legal_definitions/CheckEmbed/gpt_gpt-embedding-large_results.json",
    "results/legal_definitions/BertScore/gpt_bert.json",
    [0, 11],
    "heatmap_combined_gpt35_gpt.pdf"
)

plot_heatmap(
    "results/legal_definitions/CheckEmbed/gpt4-turbo_gpt-embedding-large_results.json",
    "results/legal_definitions/BertScore/gpt4-turbo_bert.json",
    [0, 4],
    "heatmap_combined_gpt4_gpt.pdf"
)

plot_heatmap(
    "results/legal_definitions/CheckEmbed/gpt_stella-en-1.5B-v5_results.json",
    "results/legal_definitions/BertScore/gpt_bert.json",
    [0, 11],
    "heatmap_combined_gpt35_stella.pdf"
)

plot_heatmap(
    "results/legal_definitions/CheckEmbed/gpt4-turbo_stella-en-1.5B-v5_results.json",
    "results/legal_definitions/BertScore/gpt4-turbo_bert.json",
    [0, 4],
    "heatmap_combined_gpt4_stella.pdf"
)

plot_heatmap(
    "results/legal_definitions/CheckEmbed/gpt4-o_gpt-embedding-large_results.json",
    "results/legal_definitions/BertScore/gpt4-o_bert.json",
    [0, 15],
    "heatmap_combined_gpt4o_gpt-embedding-large.pdf"
)

plot_hallucination(
    "gpt4-o",
    "ste1.5",
    "llama8b",
    "score",
)

plot_runtime(
    "results/performance_test/runtimes_results.json",
    "runtime.pdf",
    ["2_samples", "4_samples", "6_samples"],
    ["bert", "scgpt", "ce"],
    ["ste400", "ste1.5"],
    ["bert", "nli"]
)

plot_runtime(
    "results/performance_test/runtimes_results_old.json",
    "runtime_old_2-6.pdf",
    ["2_samples", "4_samples", "6_samples"],
    ["bert", "scgpt", "ce"],
    ["gpt", "sfr", "e5", "gte"],
    [None],
    True
)

plot_runtime(
    "results/performance_test/runtimes_results_old.json",
    "runtime_old_8-10.pdf",
    ["8_samples", "10_samples"],
    ["ce"],
    ["gpt", "sfr", "e5", "gte"],
    [None],
    True
)

plot_samples_accuracy(
    "results/wiki_bio",
    "plot_samples_accuracy.pdf"
)

# The following list contains the number of correct images for each prompt
# with the values stemming from manual inspection of the images in the folder
# "vision/imgs/counting_items"
plot_vision_results(
    correct_images = [
        [10, 10, 9, 8, 1 ],
        [10, 10, 8, 7, 5],
        [10, 8, 10, 8, 4],
        [9, 7, 6, 7, 6],
        [10, 10, 7, 6, 6 ],
        [10, 9, 8, 6, 5],
        [10, 9, 6, 5, 3],
        [10, 6, 4, 9, 2]
    ],
    path = "results/vision/CheckEmbed"
)

plot_wiki_bio(
    "results/wiki_bio",
    "wiki_bio.md"
)

plot_RAGTruth(
    "results/RAGTruth",
    "RAGTruth.md",
)
