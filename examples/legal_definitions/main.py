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

from CheckEmbed import language_models
from CheckEmbed import embedding_models
from CheckEmbed.plotters import BertPlot
from CheckEmbed.plotters import CheckEmbedPlot
from CheckEmbed.plotters import SelfCheckGPTPlot
from CheckEmbed.plotters import RawEmbeddingHeatPlot
from CheckEmbed.parser import Parser
from CheckEmbed.scheduler import Scheduler, StartingPoint

class CustomParser(Parser):
    """
    The CustomParser class handles the dataset parsing.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self, dataset_path: str, prompt_scheme_path: str, num_chunks: int) -> None:
        """
        Initialize the parser.

        :param dataset_path: The path to the dataset.
        :type dataset_path: str
        :param prompt_scheme_path: The path to the prompt scheme file.
        :type prompt_scheme_path: str
        :param num_chunks: The number of chunks.
        :type num_chunks: int
        """
        super().__init__(dataset_path)
        self.prompt_scheme_path = prompt_scheme_path
        self.num_chunks = num_chunks

    def prompt_generation(self, custom_inputs: Any = None) -> List[str]:
        """
        Parse the dataset and generate the prompts for the model.

        :param custom_inputs: The custom inputs to the parser. Defaults to None.
        :type custom_inputs: Any
        :return: List of prompts.
        :rtype: List[str]
        """
        # Getting the input data from the dataset
        input_data = []
        with open(self.dataset_path) as f:
            json_data = json.load(f)

        data_array = json_data['data']
        for data in data_array:
            input_data.append(data['chunk_txt'])

        # Prompts generation
        prompt_complete = None
        with open(self.prompt_scheme_path) as f:
            prompt_complete = f.read()

        prompt_initial = prompt_complete[0:prompt_complete.find('[###REPLACE WITH CONTEXT###]')]
        prompt_final = prompt_complete[prompt_complete.find('[###REPLACE WITH CONTEXT###]')+len('[###REPLACE WITH CONTEXT###]'):]

        start_index = 0
        if self.num_chunks == 1:
            start_index = 1

        # Use the input data as context inside the prompts
        prompts = []
        for i in range(start_index, len(input_data) - self.num_chunks + 1):
            prompts.append(prompt_initial + "".join(input_data[i:i+self.num_chunks]) + prompt_final)

        return prompts

    def ground_truth_extraction(self, custom_inputs: Any = None) -> List[str]:
        """
        Parse the dataset and extract the ground truth.

        :param custom_inputs: The custom inputs to the parser. Defaults to None.
        :type custom_inputs: Any
        :return: List of ground truths.
        :rtype: List[str]
        """
        ground_truth = []
        with open(self.dataset_path) as f:
            json_data = json.load(f)

        data_array = json_data['data']
        for data in data_array:
            text = ""
            for definition in data['definitions']:
                text += definition["term"] + ". " + definition["context"] + "\n"

            text = text[:-1]
            ground_truth.append(text)

        start_index = 0
        if self.num_chunks == 1:
            start_index = 1

        composite_ground_truth = []
        for i in range(start_index, len(ground_truth) - self.num_chunks + 1):
            composite_ground_truth.append("\n".join(ground_truth[i:i+self.num_chunks]))
        
        return composite_ground_truth

def start(current_dir: str, num_chunks: int = 1, start: int = StartingPoint.PROMPT, bool_magic: bool = False) -> None:
    """
    Start the main function.

    :param current_dir: The current directory.
    :type current_dir: str
    :param num_chunks: The number of chunks. Defaults to 1.
    :type num_chunks: int
    :param start: The starting point. Defaults to StartingPoint.PROMPT.
    :type start: StartingPoint
    :param bool_magic: TODO. Defaults to False.
    :type bool_magic: bool
    """

    # Config file for the LLM(s)
    config_path = os.path.join(
            current_dir,
            "../../../CheckEmbed/config.json",
        )

    # Initialize the parser and the embedder
    customParser = CustomParser("./dataset/legal_definitions.json", os.path.join(current_dir, "../prompt_scheme.txt"), num_chunks=num_chunks)

    # Initialize the language models
    gpt3 = language_models.ChatGPT(
        config_path,
        model_name = "chatgpt",
        cache = True,
    ) 

    gpt4 = language_models.ChatGPT(
        config_path,
        model_name = "chatgpt4-turbo",
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
        model_name = "Alibaba-NLP/gte-Qwen1.5-7B-instruct",
        cache = False,
        access_token = "", # Add your access token here
        batch_size=4, # it may be necessary to reduce the batch size if the GPU VRAM < 40GB
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
        budget = 30,
        parser = customParser,
        lm = [gpt4_o, gpt4, gpt3],
        embedding_lm = [embedd_large, sfrEmbeddingMistral, e5mistral7b, gteQwen157bInstruct],
        operations = [bertPlot, selfCheckGPTPlot, rawEmbeddingHeatPlot, checkEmbedPlot],
    )

    # The order of lm_names and embedding_lm_names should be the same
    # as the order of the language models and embedding language models respectively.
    scheduler.run(
        startingPoint = start,
        bertScore = False, 
        selfCheckGPT = False,
        ground_truth = True, 
        num_samples = 10, 
        lm_names = ["gpt4-o", "gpt4-turbo", "gpt"],
        embedding_lm_names = ["gpt-embedding-large", "sfr-embedding-mistral", "e5-mistral-7b-instruct", "gte-Qwen15-7B-instruct"],
        bertScore_model = "microsoft/deberta-xlarge-mnli",
        batch_size = 64, # it may be necessary to reduce the batch size if the model is too large
        device = "cuda" # or "cpu" "mps" ...
    )

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/chunk_dim_1"
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    start(current_dir, num_chunks=1, start=StartingPoint.PROMPT)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/chunk_dim_2"
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    start(current_dir, num_chunks=2, start=StartingPoint.PROMPT)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/chunk_dim_4"
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    start(current_dir, num_chunks=4, start=StartingPoint.PROMPT)
