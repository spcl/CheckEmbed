# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import logging
import os
from typing import Any, List
import json

from operation_variants import BertScoreOperation_Variant, SelfCheckGPTOperation_Variant, CheckEmbedOperation_Variant
from CheckEmbed import language_models
from CheckEmbed import embedding_models
from CheckEmbed.plotters import BertPlot
from CheckEmbed.plotters import CheckEmbedPlot
from CheckEmbed.plotters import SelfCheckGPTPlot
from CheckEmbed.plotters import RawEmbeddingHeatPlot
from CheckEmbed.parser import Parser
from CheckEmbed.scheduler import Scheduler, StartingPoint

topics_list = [
"Supernova",
"Evolution",
"Gravity",
"Quantum mechanics",
"Dark matter",
"Black hole",
"Global warming",
"Natural selection",
"Plate tectonics",
"Artificial intelligence",
"Relativity",
"Genetics",
"Big Bang theory",
"Electricity",
"Photosynthesis",
"Neuroscience",
"Climate change",
"Friction",
"Renewable energy",
"DNA",
"Democracy",
"Capitalism",
"Social media",
"Virtual reality",
"Artificial life",
"Cloning",
"Robotics",
"Space-time",
"Atomic theory",
"Consciousness"
]

class CustomParser(Parser):
    """
    The CustomParser class handles the dataset parsing.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self, dataset_path: str, prompt_scheme_path: str, list: List[str], error_number: int, final_responses_path: str) -> None:
        """
        Initialize the parser.

        :param dataset_path: The path to the dataset.
        :type dataset_path: str
        :param prompt_scheme_path: The path to the prompt scheme file.
        :type prompt_scheme_path: str
        :param list: The list of topics to be used in the prompts.
        :type list: List[str]
        :param error_number: Number of errors that the LLM is asked in the prompts to generate.
        :type error_number: int
        :param final_responses_path: The path for the hallucionations.json file.
        :type final_responses_path: str
        """
        super().__init__(dataset_path)
        self.prompt_scheme_path = prompt_scheme_path
        self.list = list
        self.error_number = error_number
        self.final_responses_path = final_responses_path

    def prompt_generation(self, custom_inputs: Any = None) -> List[str]:
        """
        Parse the dataset and generate the prompts for the model.

        :param custom_inputs: The custom inputs to the parser. Defaults to None.
        :type custom_inputs: Any
        :return: List of prompts.
        :rtype: List[str]
        """
        with open(self.prompt_scheme_path) as f:
            prompt_complete = f.read()

        # Use input data as context inside the prompts
        prompts = []
        for item in self.list:
            prompt_copy = prompt_complete
            prompts.append(prompt_copy.replace("### TOPIC ###", item).replace("### NUMBER ###", str(self.error_number)))

        return prompts

    def ground_truth_extraction(self, custom_inputs: Any = None) -> List[str]:
        """
        Parse the dataset and extract the ground truth.

        :param custom_inputs: The custom inputs to the parser. Defaults to None.
        :type custom_inputs: Any
        :return: List of ground truths.
        :rtype: List[str]
        """
        pass

    def answer_parser(self, responses: List[List[str]], custom_inputs: Any = None) -> List[List[str]]:
        """
        Parse the responses from the model.

        :param responses: The responses from the model.
        :type responses: List[List[str]]
        :param custom_inputs: The custom inputs to the parser. Defaults to None.
        :type custom_inputs: Any
        :return: The parsed responses. Output is going to be saved on a .json file using the following structure:
        {
            "data": [
                {
                    "index": "1",
                    "hallucination": "parsed_response",
                },
                {
                    "index": "2",
                    "hallucination": "parsed_response",
                },
                ...
            ]
        }
        :rtype: List[List[str]]
        """
        new_responses = []
        hallucinations = []
        for response in responses:
            new_response = []
            hallucination = []
            for res in response:
                index = res.find("### PASSAGE ###")
                new_response.append(res[index:])
                hallucination.append(res[:index])
            new_responses.append(new_response)
            hallucinations.append(hallucination)
        
        with open(self.final_responses_path + "/hallucinations.json", "w") as f:
            hallucinations_json = [{"index": i, "hallucination": hallucinations[i]} for i in range(len(hallucinations))]
            json.dump({"data": hallucinations_json}, f, indent=4)
        return new_responses

def start(current_dir: str, list: List[str], ground_truth_gen: bool = False, error_number: int = 0, start: int = StartingPoint.PROMPT) -> None:
    """
    Execute the incremental forced hallucination use case with a specific error number.

    :param current_dir: Directory path from the the script is called.
    :type current_dir: str
    :param list: The list of topics to be used in the prompts.
    :type list: List[str]
    :param ground_truth_gen: Generate ground truth. Defaults to False.
    :type ground_truth_gen: bool
    :param error_number: Number of errors that the LLM is asked in the prompts to generate. Defaults to 0.
    :type error_number: int
    :param start: Starting point indicator. Defaults to StartingPoint.PROMPT.
    :type start: int
    """
    config_path = os.path.join(
        current_dir,
        "../../../../CheckEmbed/config.json",
    )

    # Initialize the parser and the embedder
    customParser = CustomParser(
        dataset_path = current_dir,
        prompt_scheme_path = os.path.join(current_dir, "../prompt_scheme.txt" if not ground_truth_gen else "../prompt_scheme_ground_truth.txt"),
        list = list,
        error_number = error_number,
        final_responses_path = current_dir,
    )

    # Initialize the language models
    gpt3 = language_models.ChatGPT(
        config_path,
        model_name = "chatgpt",
        cache = True,
    )

    gpt4_o = language_models.ChatGPT(
        config_path,
        model_name = "chatgpt4-o",
        cache = True,
        )

    embedd_large = embedding_models.EmbeddingGPT(
        config_path,
        model_name = "gpt-embedding-large",
        cache = False,
    )

    sfrEmbeddingMistral = embedding_models.SFREmbeddingMistral(
        config_path,
        model_name = "Salesforce/SFR-Embedding-Mistral",
        cache = False,
    )

    e5mistral7b = embedding_models.E5Mistral7b(
        config_path,
        model_name = "intfloat/e5-mistral-7b-instruct",
        cache = False,
    )

    gteQwen157bInstruct = embedding_models.GteQwenInstruct(
        config_path=config_path,
        model_name= "Alibaba-NLP/gte-Qwen1.5-7B-instruct",
        cache = False,
        access_token = "", # Add your access token here
    )


    # Initialize BERTScore, SelfCheckGPT and CheckEmbedOperation operations
    bertOperation = None if ground_truth_gen else BertScoreOperation_Variant(
            os.path.join(current_dir, "BertScore"),
            os.path.join(current_dir, "../ground_truth"),
            current_dir,
        )

    selfCheckGPTOperation = None if ground_truth_gen else SelfCheckGPTOperation_Variant(
            os.path.join(current_dir, "SelfCheckGPT"),
            os.path.join(current_dir, "../ground_truth"),
            current_dir,
        )
    
    checkEmbedOperation = None if ground_truth_gen else CheckEmbedOperation_Variant(
            os.path.join(current_dir, "CheckEmbed"),
            os.path.join(current_dir, "../ground_truth/embeddings"),
            os.path.join(current_dir, "embeddings"),
        )

    # Initialize the plot operations
    bertPlot = BertPlot(
        os.path.join(current_dir, "plots", "BertScore"),
        os.path.join(current_dir, "BertScore"),
    )

    selfCheckGPTPlot = SelfCheckGPTPlot(
        os.path.join(current_dir, "plots", "SelfCheckGPT"),
        os.path.join(current_dir, "SelfCheckGPT"),
    )

    rawEmbeddingHeatPlot = RawEmbeddingHeatPlot(
        os.path.join(current_dir, "plots", "CheckEmbed"),
        os.path.join(current_dir, "embeddings"),
    )

    checkEmbedPlot = CheckEmbedPlot(
        os.path.join(current_dir, "plots", "CheckEmbed"),
        os.path.join(current_dir, "CheckEmbed"),
    )

    # Initialize the scheduler
    scheduler = Scheduler(
        current_dir,
        logging_level = logging.DEBUG,
        budget = 10,
        parser = customParser,
        lm = [gpt4_o, gpt3],
        embedding_lm = [embedd_large, sfrEmbeddingMistral, e5mistral7b, gteQwen157bInstruct],
        operations = [bertPlot, selfCheckGPTPlot, rawEmbeddingHeatPlot, checkEmbedPlot] if not ground_truth_gen else [],
        bertScoreOperation = bertOperation,
        selfCheckGPTOperation = selfCheckGPTOperation,
        checkEmbedOperation = checkEmbedOperation,
    )

    # The order of lm_names and embedding_lm_names should be the same 
    # as the order of the language models and embedding language models respectively.
    scheduler.run(
        startingPoint = start,
        bertScore = True,
        selfCheckGPT = True,
        ground_truth = False,
        spacy_separator = True,
        num_samples = 10,
        lm_names = ["gpt4-o", "gpt"],
        embedding_lm_names = ["gpt-embedding-large", "sfr-embedding-mistral", "e5-mistral-7b", "gte-qwen-1.5-7b-instruct"],
        bertScore_model = "microsoft/deberta-xlarge-mnli",
        batch_size = 64, # it may be necessary to reduce the batch size if the model is too large
        device = "cuda" # or "cpu" "mps" ...
    )

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/ground_truth"
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    start(current_dir, topics_list, ground_truth_gen=True, error_number=0)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_1"
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    start(current_dir, topics_list, ground_truth_gen=False, error_number=1)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_2"
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    start(current_dir, topics_list, ground_truth_gen=False, error_number=2)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_3"
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    start(current_dir, topics_list, ground_truth_gen=False, error_number=3)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_4"
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    start(current_dir, topics_list, ground_truth_gen=False, error_number=4)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_5"
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    start(current_dir, topics_list, ground_truth_gen=False, error_number=5)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_6"
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    start(current_dir, topics_list, ground_truth_gen=False, error_number=6)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_7"
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    start(current_dir, topics_list, ground_truth_gen=False, error_number=7)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_8"
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    start(current_dir, topics_list, ground_truth_gen=False, error_number=8)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_9"
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    start(current_dir, topics_list, ground_truth_gen=False, error_number=9)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_10"
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    start(current_dir, topics_list, ground_truth_gen=False, error_number=10)
