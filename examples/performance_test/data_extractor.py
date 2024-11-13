# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import os
import json
import re

from typing import List, Dict, Tuple

steps = ["Embedding generation", "BERTScore operation", "SelfCheckGPT_BertScore operation", "SelfCheckGPT_NLI operation", "SelfCheckGPT operation", "CheckEmbed operation", "Operations"]

def extract_embeddings(lines: List[str], i: int) -> Tuple[Dict[str, Dict[str, float]], int]:
    """
    Extract the runtimes for the embedding generation.

    :param lines: Lines of the input files.
    :type lines:  List[str]
    :param i: Index of the lines that are already processed.
    :type i: int
    :return: Tuple of the extracted runtimes and the index of the already processed lines.
    :rtype: Tuple[Dict[str, Dict[str, float]], int]
    """
    embeddings = {}
    emb_pattern = r"(?<=Embedding model: )\S+"
    runtime_pattern = r"LM (\S+): ([\d.]+) seconds"

    i += 1
    while i < len(lines) and ("Total time" not in lines[i] or "- Total time" in lines[i]):
        result = re.search(emb_pattern, lines[i])
        if result:
            runtimes = {}
            while i < len(lines) and "- Total time" not in lines[i]:
                search_res = re.search(runtime_pattern, lines[i])
                if search_res:
                    runtimes[search_res.group(1)] = float(search_res.group(2))

                i += 1
            
            embeddings[result.group()] = runtimes 
        i += 1 

    print("Embeddings extracted successfully!")
    
    return embeddings, i


def extract_bertscore(lines: List[str], i: int) -> Tuple[Dict[str, float], int]:
    """
    Extract the runtimes for the BERTScore operation.

    :param lines: Lines of the input files.
    :type lines:  List[str]
    :param i: Index of the lines that are already processed.
    :type i: int
    :return: Tuple of the extracted runtimes and the index of the already processed lines.
    :rtype: Tuple[Dict[str, float], int]
    """

    bertscore = {}
    runtime_pattern = r"- Time for (\S+): ([\d.]+)"

    i += 1
    while i < len(lines) and "Total time" not in lines[i]:
        search_res = re.search(runtime_pattern, lines[i])
        if search_res:
            bertscore[search_res.group(1)] = float(search_res.group(2))

        i += 1

    print("BERTScore extracted successfully!")

    return bertscore, i


def extract_selfcheckgpt(lines: List[str], i: int) -> Tuple[Dict[str, float], int]:
    """
    Extract the runtimes for the SelfCheckGPT operation.

    :param lines: Lines of the input files.
    :type lines:  List[str]
    :param i: Index of the lines that are already processed.
    :type i: int
    :return: Tuple of the extracted runtimes and the index of the already processed lines.
    :rtype: Tuple[Dict[str, float], int]
    """

    selfcheckgpt = {}
    runtime_pattern = r"- Time for (\S+): ([\d.]+)"

    i += 1
    while i < len(lines) and "Total time" not in lines[i]:
        search_res = re.search(runtime_pattern, lines[i])
        if search_res:
            selfcheckgpt[search_res.group(1)] = float(search_res.group(2))

        i += 1

    print("SelfCheckGPT extracted successfully!")

    return selfcheckgpt, i


def extract_checkembed(lines: List[str], i: int) -> Tuple[Dict[str, Dict[str, float]], int]:
    """
    Extract the runtimes for the CheckEmbed operation.

    :param lines: Lines of the input files.
    :type lines:  List[str]
    :param i: Index of the lines that are already processed.
    :type i: int
    :return: Tuple of the extracted runtimes and the index of the already processed lines.
    :rtype: Tuple[Dict[str, Dict[str, float]], int]
    """

    checkembed = {}
    runtime_pattern = r"- Time for (\S+)\s+(\S+): ([\d.]+)"

    i += 1
    while i < len(lines) and "\n" != lines[i]:
        search_res = re.search(runtime_pattern, lines[i])
        if search_res:
            if checkembed.get(search_res.group(2)):
                checkembed[search_res.group(2)].update({search_res.group(1): float(search_res.group(3))})
            else:
                checkembed[search_res.group(2)] = {search_res.group(1): float(search_res.group(3))}

        i += 1

    def custom_sort_key(item):
        key, _ = item
        try:
            # Try to convert the key to an integer
            return (0, int(key))
        except ValueError:
            # If it fails, it's a string
            return (1, key)
        
    # Reorder alphabetically and numerically the internal dictionary
    for key in checkembed:
        checkembed[key] = dict(sorted(checkembed[key].items(), key=custom_sort_key))

    print("CheckEmbed extracted successfully!")

    return checkembed, i

def extract_operation(lines: List[str], i: int) -> Tuple[Dict[str, float], int]:
    """
    Generic function interface to extract the runtimes of an operation.

    Customize to your needs.

    :param lines: Lines of the input files.
    :type lines:  List[str]
    :param i: Index of the lines that are already processed.
    :type i: int
    :return: Tuple of the extracted runtimes and the index of the already processed lines.
    :rtype: Tuple[Dict[str, float], int]
    """
    return {}, i+1


def extract(
        samples: List[str],
        runtime_dir: str, 
        runtime_file_name: str, 
        result_dir: str, 
        result_file_name: str
    ) -> None:
    """
    Extract the runtimes from the files of the different sample sizes.

    :param samples: List of names of the sample size directories.
    :type samples: List[str]
    :param runtime_dir: Path to the runtime directory, the input directory.
    :type runtime_dir: str
    :param runtime_file_name: Name of the input files.
    :type runtime_file_name: str
    :param result_dir: Path to the output directory.
    :type result_dir: str
    :param result_file_name: Name of the output file.
    :type result_file_name: str
    """

    complete_results = {}
    for sample in samples:
        sample_results = {}

        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), sample, runtime_dir, runtime_file_name), "r") as f:
            lines = f.readlines()

        for i in range(len(lines)):
            for step in steps:
                if step in lines[i]:
                    if step == "Embedding generation":
                        sample_results["embedding"], i = extract_embeddings(lines, i)
                    elif step == "BERTScore operation":
                        sample_results["bertscore"], i = extract_bertscore(lines, i)
                    elif step == "SelfCheckGPT_BertScore operation":
                        sample_results["selfcheckgpt_bertscore"], i = extract_selfcheckgpt(lines, i)
                    elif step == "SelfCheckGPT_NLI operation":
                        sample_results["selfcheckgpt_nli"], i = extract_selfcheckgpt(lines, i)
                    elif step == "SelfCheckGPT operation":
                        sample_results["selfcheckgpt"], i = extract_selfcheckgpt(lines, i)
                    elif step == "CheckEmbed operation":
                        sample_results["checkembed"], i = extract_checkembed(lines, i)
                    elif step == "Operations":
                        sample_results["operations"], i = extract_operation(lines, i)
        
        complete_results[sample] = sample_results
        print(f"Results extracted for {sample} successfully!")
    
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), result_dir, result_file_name), "w") as f:
        json.dump(complete_results, f, indent=4)
    
    print("Results extracted successfully!")


if __name__ == "__main__":
    samples = ["2_samples", "4_samples", "6_samples", "8_samples", "10_samples"]

    runtime_dir = "runtimes"
    runtime_file_name = "performance_log.log"

    result_dir = "."
    result_file_name = "runtimes_results.json"

    extract(samples, runtime_dir, runtime_file_name, result_dir, result_file_name)
