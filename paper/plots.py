# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main authors:
# Patrick Iff
# Robert Gerstenberger
#
# contributions:
# Lorenzo Paleari


import csv
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import numpy as np

from decimal import Decimal


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
    for entry in data["data"]:
        metric_map_score = {"bert" : "frobenius_norm", "scgpt" : "passage_score", "ce" : "frob_norm_cosine_sim", "ce_got" : "frob_norm_cosine_sim"}
        metric_map_std_dev = {"bert" : "std_dev", "scgpt" : None, "ce" : "std_dev_cosine_sim", "ce_got" : "std_dev_cosine_sim"}
        metric_map_map = {"score" : metric_map_score, "stddev" : metric_map_std_dev}
        key = metric_map_map[metric][method]
        if key != None and key in entry:
            results.append(entry[key])
    return results


# written by Patrick Iff
def read_file(error, model, method, emb_model, metric):
    dir1 = "incremental_forced_hallucination"
    dir2 = ("error_%d" % error) if type(error) == int else error
    dir3 = {"bert" : "BertScore", "scgpt" : "SelfCheckGPT", "ce" : "CheckEmbed", "ce_got" : "CheckEmbed_got"}[method]
    emb_model = {"gpt" : "gpt-embedding-large_results", "sfr" : "sfr-embedding-mistral_results", "e5" : "e5-mistral-7b-instruct_results"}[emb_model]
    file = model + "_" + {"bert" : "bert", "scgpt" : "gptselfcheck", "ce" : emb_model, "ce_got" : emb_model}[method] + ".json"
    path = "results/%s/%s/%s/%s" % (dir1, dir2, dir3, file)
    # Hack
    if error == "got_data" and method == "ce_got":
        path = path.replace("CheckEmbed_got", "CheckEmbed")
    return read_file_base(path, method, metric)


# written by Patrick Iff
def read_all_files(model, emb_model, metric):
    errors = ["got_data"] + list(range(1, 11))
    methods = ["bert", "scgpt", "ce_got"]
    data = {}
    for error in errors:
        data[error] = {}
        for method in methods:
            sub_data =  read_file(error, model, method, emb_model, metric)
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
    dir2 = {"bert" : "BertScore", "scgpt" : "SelfCheckGPT", "ce" : "CheckEmbed"}[method]
    emb_model = {"gpt" : "gpt-embedding-large_results", "sfr" : "sfr-embedding-mistral_results", "e5" : "e5-mistral-7b-instruct_results", "gte" : "gte-Qwen15-7B-instruct_results" , None : ""}[emb_model]
    file = model + "_" + {"bert" : "bert", "scgpt" : "gptselfcheck", "ce" : emb_model}[method] + ".json"
    path = "results/%s/%s/%s" % (dir1, dir2, file)
    data = read_json_file(path)
    results = []
    for entry in data["data"]:
        score = entry[{"bert" : "frobenius_norm", "scgpt" : "passage_score", "ce" : "frob_norm_cosine_sim"}[method]]
        results.append(score)
    return results


# Read all files containing results
# written by Patrick Iff
def read_all_description_files(mode):
    types = ["different","similar"]
    methods = ["bert","scgpt","ce"]
    models = ["gpt","gpt4-turbo","gpt4-o"]
    embedding_models = ["gpt","sfr","e5","gte"]
    data = {}
    for model in models:
        data[model] = {}
        for method in methods:
            emb_models = embedding_models if method == "ce" else [None]
            for emb_model in emb_models:
                method_label = method + ("_" + emb_model if emb_model is not None else "")
                data[model][method_label] = {}
                for typ in types:
                    data[model][method_label][typ] = read_description_file(typ, method, model, mode, emb_model)
    return data


# Create violin plot
# written by Patrick Iff
def plot_description(mode):
    # Config
    colors = {"different" : "#990000", "similar" : "#009900"}
    method_labels = {"bert" : "BERTScore", "scgpt" : "SelfCheckGPT", "ce" : "CheckEmbed", "ce_gpt" : "CheckEmbed (GPT)", "ce_sfr" : "CheckEmbed (SFR)", "ce_e5" : "CheckEmbed (E5)", "ce_gte" : "CheckEmbed (GTE)"}
    model_labels = {"gpt4-o" : "GPT-4o", "gpt4-turbo" : "GPT-4-turbo", "gpt" : "GPT-3.5"}
    # Read data
    data = read_all_description_files(mode)
    # Create plot
    (fig, ax) = plt.subplots(1, 3, figsize=(9, 3.5))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.925, bottom=0.25, wspace=0.05)
    # Iterate through data
    for (i, model) in enumerate(data.keys()):
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
                legend_patches.add((col, typ.capitalize() + " Replies"))
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
        ax[i].legend(handles=list(legend_patches), loc='lower right', fontsize=9)
    # Save plot
    plt.savefig("plot_eval_violins_gpt_%s.pdf" % mode)


# written by Patrick Iff
def plot_hallucination(model, emb_model, metric):
    # Config
    colors = {"bert" : "#999900", "scgpt" : "#990099", "ce" : "#000099", "ce_got" : "#009999"}
    method_labels = {"GOT" : "GOT", "bert" : "BERTScore", "scgpt" : "SelfCheckGPT", "ce" : "CheckEmbed", "ce_got" : "CheckEmbed"}
    model_labels = {"gpt4-o" : "GPT-4o", "gpt4-turbo" : "GPT-4-turbo", "gpt" : "GPT-3.5"}
    emb_model_labels = {"gpt" : "GPT-3.5", "sfr" : "SFR", "e5" : "E5"}
    # Read data
    data = read_all_files(model, emb_model, metric)
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
    ax.set_xlabel(" " * 17 + "Error")
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

    plt.savefig("plot_halucinate_%s_%s_%s.pdf" % (model, emb_model, metric))


# written by Patrick Iff
def plot_runtimes():
    # Config
    method_labels = {"bert" : "BERTScore", "scgpt" : "SelfCheckGPT", "ce" : "CheckEmbed", "ce_gpt" : "CheckEmbed (GPT)", "ce_sfr" : "CheckEmbed (SFR)", "ce_e4" : "CheckEmbed (E5)", "ce_gte" : "CheckEmbed (GTE)"}
    # Read a csv file into a python list
    data = {}
    with open('results/runtime_data.csv', 'r') as file:
        reader = csv.reader(file)
        keys = next(reader)
        for row in reader:
            for i, key in enumerate(keys):
                if key not in data:
                    data[key] = []
                data[key].append(row[i])
    # Sort data
    methods = [method_labels[x] for x in data['method']]
    total_time = [float(x) for x in data['total']]
    # Create plot
    fig, ax = plt.subplots(1,1, figsize=(2.5, 3.5))
    fig.subplots_adjust(left=0.25, right=0.975, top=0.975, bottom=0.3)
    # Plot the data with stacking
    ax.bar(methods, total_time, label='Total', color='#000000', zorder=3)
    ax.grid(axis = "y", zorder=0)
    # Add labels and title
    ax.set_ylabel('Runtime [s]')
    ax.set_xticklabels(methods, rotation=35, ha='right')
    # Save the plot
    plt.savefig('runtime_plot.pdf')


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

    axs[0, 0].set_title(f"CheckEmbed", fontsize=fig_fontsize+8, weight='bold', color='#008000')
    axs[0, 1].set_title(f"BERTScore", fontsize=fig_fontsize+8, weight='bold', color='#800000')

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
                axs[row, col].set_xlabel(f"LLM Reply ID or Ground-Truth (GT)", fontsize=fig_fontsize)
            else:
                axs[row, col].set_xticks([])
            if col == 0:
                axs[row, col].set_yticks(np.arange(len(cosine_similarity_matrix)))
                axs[row, col].set_yticklabels(tick_labels, fontsize=fig_fontsize)
                axs[row, col].set_ylabel(f"LLM Reply ID or Ground-Truth (GT)", fontsize=fig_fontsize)
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
    "heatmap_combined_gpt35.pdf"
)

plot_heatmap(
    "results/legal_definitions/CheckEmbed/gpt4-turbo_gpt-embedding-large_results.json",
    "results/legal_definitions/BertScore/gpt4-turbo_bert.json",
    [0, 4],
    "heatmap_combined_gpt4.pdf"
)

plot_hallucination(
    "gpt4-o",
    "gpt",
    "score"
)

plot_runtimes()
