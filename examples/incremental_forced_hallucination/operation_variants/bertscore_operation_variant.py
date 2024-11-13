# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import logging
import os
import json

import bert_score
import numpy as np
from tqdm import tqdm
from typing import Any

from CheckEmbed.operations import Operation
from CheckEmbed.utility import capture_specific_stderr, frobenius_norm

class BertScoreOperation_Variant(Operation):
    """
    Operation that computes the BertScore between the reference and the sample embeddings.

    Inherits from the Operation class and implements its abstract methods.
    """

    def __init__(self, result_dir_path: str, ground_truth_dir_path: str, sample_dir_path: str) -> None:
        """
        Initialize the operation.

        :param result_dir_path: The path to the directory where the results will be stored.
        :type result_dir_path: str
        :param ground_truth_dir_path: The path to the directory where the ground truth samples are stored.
        :type ground_truth_dir_path: str
        :param sample_dir_path: The path to the directory where the samples are stored.
        :type sample_dir_path: str
        """
        super().__init__(result_dir_path)
        self.sample_dir_path = sample_dir_path
        self.ground_truth = ground_truth_dir_path

    def execute(self, custom_inputs: Any) -> Any:
        """
        Execute the operation on the embeddings/samples.

        :param custom_inputs: The custom inputs for the operation.
        :type custom_inputs: any
        """

        print("\n\nRunning BertScore operation.")
        
        # Initialize logging
        logging.basicConfig(
            filename=os.path.join(self.result_dir_path, "log.log"),
            filemode="w",
            format="%(name)s - %(levelname)s - %(message)s",
            level=custom_inputs["logging_level"],
        )

        # Run BertScore for every pair of language model and samples
        for lm_name in (pbar := tqdm(custom_inputs["lm_names"], desc="Language Models", leave=True)):
            pbar.set_postfix_str(f"{lm_name}")
            logging.info(f"Loading responses from {lm_name}.")
            samples = []

            # Load samples from the language model
            with open(os.path.join(self.sample_dir_path, f"{lm_name}_samples.json")) as f:
                responses = json.load(f)

            for index, response in enumerate(responses["data"]):
                samples.append(response["samples"])
                logging.debug(f"Sample {index}: {samples[index]}")

            logging.info("Loaded samples.")

            # load second set of samples that will be ground_truth data
            ground_truth_list = []
            with open(os.path.join(self.ground_truth, f"{lm_name}_samples.json")) as f:
                responses = json.load(f)

            for index, response in enumerate(responses["data"]):
                ground_truth_list.append(response["samples"])
                logging.debug(f"ground_truth {index}: {ground_truth_list[index]}")
            
            # For every prompt compare every sample with every other sample
            logging.info(f"Running BertScore for {lm_name}.")

            same_samples = []
            for sample in samples:
                same_s = []
                for i in range(len(sample)):
                    temp = []
                    for j in range(len(sample)):
                        temp.append(sample[i])
                    same_s.append(temp)
                same_samples.append(same_s)

            results = []
            for ground_truth, same_sample in tqdm(zip(ground_truth_list, same_samples), total=len(samples), desc="Prompts", leave=False):
                result = []
                for s in tqdm(same_sample, desc="Samples", leave=False):
                    target_string = ["Warning: Empty reference sentence detected; setting raw BERTScores to 0.","Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']", "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."]
                    with capture_specific_stderr(target_string):
                        result.append(bert_score.score(
                            ground_truth, s, model_type=custom_inputs["model_type"],
                            batch_size=custom_inputs["batch_size"], device=custom_inputs["device"],
                            lang="en", verbose=False,
                            rescale_with_baseline=True,
                        )[2].tolist())
                results.append(result)
                logging.debug(f"Results: {result}")
            
            logging.info(f"Finished running BertScore for {lm_name}.")

            # Fix the results that are less than -1
            for index, result in enumerate(results):
                temp_res = np.zeros((len(result), len(result[0])))
                for i in range(temp_res.shape[0]):
                    for j in range(temp_res.shape[1]):
                        if temp_res[i][j] < -1:
                            temp_res[i][j] = -1
                        else:
                            temp_res[i][j] = result[i][j]
                results[index] = temp_res

            frobenius_norms = [frobenius_norm(result) for result in results]
            std_devs = [np.std(result) for result in results]

            # Store results
            with open(os.path.join(self.result_dir_path, f"{lm_name}_bert.json"), "w") as f:
                results_json = [{
                    "index": i,
                    "result": result.tolist(),
                    "frobenius_norm": frob_norm,
                    "std_dev": std_dev
                } for i, result, frob_norm, std_dev in zip(range(len(results)), results, frobenius_norms, std_devs)]
                json.dump({"data": results_json}, f, indent=4)

            logging.info(f"Saved results for {lm_name}.")
