# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# contains code from the SelfCheckGPT framework released under the MIT
# license:
# https://github.com/potsawee/selfcheckgpt
#
# main author: Lorenzo Paleari

import logging
import os
import json
import spacy
import torch
import numpy as np
import bert_score

from tqdm import tqdm
from typing import List, Any, Tuple
from timeit import default_timer as timer

from CheckEmbed.operations import Operation
from CheckEmbed.utility import capture_specific_stderr

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI


# This class is copied from the SelfCheckGPT GitHub repository.
# https://github.com/potsawee/selfcheckgpt/blob/main/selfcheckgpt/modeling_selfcheck.py
#
# Utils used in the original code have been moved inside the class
# https://github.com/potsawee/selfcheckgpt/blob/main/selfcheckgpt/utils.py
#
# Released under a MIT license.
#
# modifications by Lorenzo Paleari:
# - BertScore parameter updates
# - added utils functions inside the class
# - bug fixes

class SelfCheckBERTScore:
    """
    SelfCheckGPT (BERTScore variant): Checking LLM's text against its own sampled texts via BERTScore (against best-matched sampled sentence)
    """
    def __init__(self, device: str, batch_size: str, default_model="en", rescale_with_baseline=True):
        """
        :default_model: model for BERTScore
        :rescale_with_baseline:
            - whether or not to rescale the score. If False, the values of BERTScore will be very high
            - this issue was observed and later added to the BERTScore package,
            - see https://github.com/Tiiiger/bert_score/blob/master/journal/rescale_baseline.md
        """
        # device and batch_size have been added to the original code to allow for more flexibility
        self.nlp = spacy.load("en_core_web_sm")
        self.default_model = default_model # en => roberta-large
        self.rescale_with_baseline = rescale_with_baseline
        self.device = device
        self.batch_size = batch_size
        # print("SelfCheck-BERTScore initialized") Removed from the original code

    # SelfCheck - BERTScore utils
    def expand_list1(self, mylist, num):
        expanded = []
        for x in mylist:
            for _ in range(num):
                expanded.append(x)
        return expanded

    def expand_list2(self, mylist, num):
        expanded = []
        for _ in range(num):
            for x in mylist:
                expanded.append(x)
        return expanded

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :return sent_scores: sentence-level score which is 1.0 - bertscore
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        bertscore_array = np.zeros((num_sentences, num_samples))
        for s in range(num_samples):
            sample_passage = sampled_passages[s]
            sentences_sample = [sent for sent in self.nlp(sample_passage).sents] # List[spacy.tokens.span.Span]
            # NEXT LINE IS BUGGED, IT CAN CAUSE THE ARRAY TO BE EMPTY AND BERTSCORE TO CRASH
            sentences_sample = [sent.text.strip() for sent in sentences_sample if len(sent) > 3]
            num_sentences_sample  = len(sentences_sample)

            refs  = self.expand_list1(sentences, num_sentences_sample) # r1,r1,r1,....
            cands = self.expand_list2(sentences_sample, num_sentences) # s1,s2,s3,...

            # Added to original code to fix bug - if there are no references, skip the iteration
            if (len(refs) == 0):
                continue

            with capture_specific_stderr(): # Added to original code to suppress warnings
                P, R, F1 = bert_score.score(
                    cands, refs,
                    device=self.device, batch_size=self.batch_size,
                    lang=self.default_model, verbose=False,
                    rescale_with_baseline=self.rescale_with_baseline,
                )

            F1_arr = F1.reshape(num_sentences, num_sentences_sample)
            F1_arr_max_axis1 = F1_arr.max(axis=1).values
            F1_arr_max_axis1 = F1_arr_max_axis1.numpy()

            bertscore_array[:,s] = F1_arr_max_axis1

        bertscore_mean_per_sent = bertscore_array.mean(axis=-1)
        one_minus_bertscore_mean_per_sent = 1.0 - bertscore_mean_per_sent
        return one_minus_bertscore_mean_per_sent


class SelfCheckGPT_Operation_Variant(Operation):
    """
    Base class for the SelfCheckGPT operations.
    """

    def __init__(self, result_dir_path: str, ground_truth_dir: str, sample_dir_path: str) -> None:
        """
        Initialize the operation.

        :param result_dir_path: The path to the directory where the results will be stored.
        :type result_dir_path: str
        :param ground_truth_dir: The path to the directory where the ground truth is stored.
        :type ground_truth_dir: str
        :param sample_dir_path: The path to the directory where the samples are stored.
        :type sample_dir_path: str
        """
        super().__init__(result_dir_path)
        self.ground_truth_dir = ground_truth_dir
        self.sample_dir_path = sample_dir_path

    def execute(self, custom_inputs: Any) -> None:
        """
        Execute the operation on the embeddings/samples.

        :param custom_inputs: The custom inputs for the operation.
        :type custom_inputs: Any
        """

        print("Running SelfCheckGPT operation.")
        time_performance = custom_inputs["time_performance"]
        
        # Initialize logging
        logging.basicConfig(
            filename=os.path.join(self.result_dir_path, "log.log"),
            filemode="w",
            format="%(name)s - %(levelname)s - %(message)s",
            level=custom_inputs["logging_level"],
        )

        # Initialize SelfCheckGPT class and return name.
        selfcheck_instance, name = self.instance_selfcheckgpt(custom_inputs)
        logging.info(f"SelfCheckGPT with {name} initialized.")

        if time_performance:
            with open(os.path.join(self.sample_dir_path, "runtimes", "performance_log.log"), "a") as f:
                f.write(f"\n\nSelfCheckGPT_{name} operation\n")

        # For every language model response file run SelfCheckGPT with BertScore
        performance_times = []
        for lm_name in (pbar := tqdm(custom_inputs["lm_names"], desc="Language Models", leave=True)):
            pbar.set_postfix_str(f"{lm_name}")
            logging.info(f"Loading responses from {lm_name}.")
            reference_texts = []
            samples = []

            start = timer() if time_performance else None

            # Load samples from the language model
            with open(os.path.join(self.ground_truth_dir, f"{lm_name}_samples.json")) as f:
                responses = json.load(f)

            nlp = spacy.load("en_core_web_sm")

            # Using first sample of each prompt as reference text
            for index, response in enumerate(responses["data"]):
                if custom_inputs["spacy"]:
                    reference_texts.append([sent.text.strip() for sent in nlp(response["samples"][0]).sents])
                else:
                    reference_texts.append([sentence.strip() for sentence in response["samples"][0].split("\n")])
                logging.debug(f"Reference text {index}: {reference_texts[index]}")

            logging.info("Loaded reference texts.")

            # Using the rest of the samples as samples
            with open(os.path.join(self.sample_dir_path, f"{lm_name}_samples.json")) as f:
                responses = json.load(f)

            for index, response in enumerate(responses["data"]):
                samples.append(response["samples"])
                logging.debug(f"Sample {index}: {samples[index]}")

            logging.info("Loaded samples.")

            # Run SelfCheckGPT with BertScore
            logging.info(f"Running SelfCheckGPT_{name} for {lm_name}.")

            results = []
            for reference_text, sample in tqdm(zip(reference_texts, samples), desc="Prompts", leave=False, total=len(reference_texts)):
                clean_sample = [s for s in sample if s != ""] # Remove empty strings
                clean_reference_text = [s for s in reference_text if s != ""] # Remove empty strings
                if clean_reference_text == [] or clean_sample == []:
                    results.append([])
                    continue
                results.append([res for res in selfcheck_instance.predict(sentences=clean_reference_text, sampled_passages=clean_sample)])
                logging.debug(f"Results: {results[-1]}")

            # Invert results - results are given with 0 less hallucination, 1 more hallucination
            results = [[0.0 if res > 1.0 else 1 - res for res in result] for result in results]
            passage_scores = [sum(result) / len(result) if len(result) > 0 else 0.0 for result in results]
            std_devs = [np.std(result) if len(result) > 0 else 0.0 for result in results]
            
            logging.info(f"Finished running SelfCheckGPT_{name} for {lm_name}.")
            end = timer() if time_performance else None
            if time_performance:
                performance_times.append(end - start)
                with open(os.path.join(self.sample_dir_path, "runtimes", "performance_log.log"), "a") as f:
                    f.write(f"\t - Time for {lm_name}: {end - start}\n")

            # Store results
            with open(os.path.join(self.result_dir_path, f"{lm_name}_selfcheckgpt_{name}.json"), "w") as f:
                results_json = [{
                    "index": i,
                    "result": result,
                    "passage_score": passage_score,
                    "std_dev": std_dev,
                    } for i, (result, passage_score, std_dev) in enumerate(zip(results, passage_scores, std_devs))]
                json.dump({"data": results_json}, f, indent=4)

            logging.info(f"Saved results for {lm_name}.")

        if time_performance:
            with open(os.path.join(self.sample_dir_path, "runtimes", "performance_log.log"), "a") as f:
                f.write(f"\n\tTotal time: {sum(performance_times)}\n")

    def instance_selfcheckgpt(self, custom_inputs: Any) -> Tuple[Operation, str]:
        """
        Initialize the SelfCheckGPT operation with the given custom inputs.

        :param custom_inputs: The custom inputs for the operation.
        :type custom_inputs: Any
        """
        raise NotImplementedError("The init_selfcheckgpt method must be implemented.")

class SelfCheckGPT_BERT_Operation_Variant(SelfCheckGPT_Operation_Variant):
    """
    Operation that computes the SelfCheckGPT score using Bert for the samples.

    Inherits from the Operation class and implements its abstract methods.
    """

    def __init__(self, result_dir_path: str, ground_truth_dir: str, sample_dir_path: str) -> None:
        """
        Initialize the operation.

        :param result_dir_path: The path to the directory where the results will be stored.
        :type result_dir_path: str
        :param ground_truth_dir: The path to the directory where the ground truth is stored.
        :type ground_truth_dir: str
        :param sample_dir_path: The path to the directory where the samples are stored.
        :type sample_dir_path: str
        """
        super().__init__(result_dir_path, ground_truth_dir, sample_dir_path)

    def instance_selfcheckgpt(self, custom_inputs: Any) -> Tuple[Operation, str]:
        """
        Initialize the SelfCheckGPT operation with the given custom inputs.

        :param custom_inputs: The custom inputs for the operation.
        :type custom_inputs: Any
        
        :return: The initialized SelfCheckGPT operation and its name.
        :rtype: Tuple[Operation, str]
        """

        # Initialize SelfCheckGPT with BertScore
        selfcheck_bertscore = SelfCheckBERTScore(device=custom_inputs["device"], batch_size=custom_inputs["batch_size"], rescale_with_baseline=True)
        return selfcheck_bertscore, "BertScore"


class SelfCheckGPT_NLI_Operation_Variant(SelfCheckGPT_Operation_Variant):
    """
    Operation that computes the SelfCheckGPT score using NLI for the samples.

    Inherits from the Operation class and implements its abstract methods.
    """

    def __init__(self, result_dir_path: str, ground_truth_dir: str, sample_dir_path: str) -> None:
        """
        Initialize the operation.

        :param result_dir_path: The path to the directory where the results will be stored.
        :type result_dir_path: str
        :param ground_truth_dir: The path to the directory where the ground truth is stored.
        :type ground_truth_dir: str
        :param sample_dir_path: The path to the directory where the samples are stored.
        :type sample_dir_path: str
        """
        super().__init__(result_dir_path, ground_truth_dir, sample_dir_path)

    def instance_selfcheckgpt(self, custom_inputs: Any) -> Tuple[Operation, str]:
        """
        Initialize the SelfCheckGPT operation with the given custom inputs.

        :param custom_inputs: The custom inputs for the operation.
        :type custom_inputs: Any

        :return: The initialized SelfCheckGPT operation and its name.
        :rtype: Tuple[Operation, str]
        """
        # Initialize SelfCheckGPT with NLI
        device = torch.device(custom_inputs["device"])
        selfcheck_nli = SelfCheckNLI(device=device)
        logging.info("SelfCheckGPT with NLI initialized.")

        return selfcheck_nli, "NLI"