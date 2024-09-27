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

precise_topics = [
"Old, rusted bicycle leaning against a weathered fence",
"Brightly colored hot air balloons rising at dawn",
"Abandoned, overgrown amusement park",
"Cozy, candle-lit cottage in the woods",
"Bustling farmers' market on a sunny morning",
"Quiet, fog-covered cemetery",
"Antique typewriter on a wooden desk",
"Snow-capped mountains reflecting in a pristine lake",
"Cracked leather armchair in a dimly lit library",
"Vintage record player spinning a classic vinyl",
"Sunflower field swaying in the summer breeze",
"City skyline at night with lights reflecting on the river",
"Deserted, windswept beach at sunset",
"Steaming cup of coffee on a rainy window sill",
"Dilapidated barn surrounded by golden wheat fields",
"Intricately carved wooden chess set mid-game",
"Tropical rainforest teeming with vibrant wildlife",
"Quaint village square with cobblestone streets",
"Icy mountain glacier under a clear blue sky",
"Sleek, modern skyscraper towering over the city",
"Crashing waves against jagged coastal cliffs",
"Autumn leaves crunching underfoot in a dense forest",
"Neon signs illuminating a bustling urban street",
"Old, weather-beaten ship docked in a quiet harbor",
"Serene, lotus-covered pond in a tranquil garden",
"Majestic eagle soaring high above the canyon",
"Rustic, stone fireplace crackling with a warm fire",
"Dense, foggy moorland with rolling hills",
"Starry night sky over a silent desert",
"Crystal chandelier hanging in an opulent ballroom",
"Abandoned train station covered in graffiti",
"Snow-covered village during a silent, starry night",
"Ancient, moss-covered stone bridge over a bubbling brook",
"Golden retriever playing fetch in a sunny park",
"Bustling open-air café in a European city",
"Broken-down carousel with faded paint",
"Sunset over a tranquil lake with fishing boats",
"Children playing in a splash fountain on a hot day",
"Quiet, snow-blanketed forest",
"Family gathered around a barbecue in a backyard",
"Lush vineyard with ripe grapes hanging from the vines",
"Rain-soaked city streets glistening under streetlights",
"Sheep grazing on a hillside under a clear sky",
"Street musician playing a violin in a busy square",
"Balloons tied to a park bench swaying in the breeze",
"Old bookshop with shelves crammed full of books",
"Kite flying high above a windy hilltop",
"Cobbled alleyway with ivy climbing the walls",
"Ducklings following their mother across a pond",
"Majestic castle perched on a rocky cliff"
]

general_topics = [
"Vintage cars",
"Mountaintop views",
"Old libraries",
"Jazz music in a dimly lit club",
"Handwritten letters",
"Art deco architecture",
"Street food markets",
"Historic battlefields",
"Train journeys through countryside",
"Surfing at sunrise",
"Hidden speakeasies",
"Ancient ruins",
"Traditional tea ceremonies",
"Opera houses",
"Mid-century modern furniture",
"Botanical gardens",
"Old-fashioned barbershops",
"Film noir movies",
"Victorian mansions",
"Renaissance paintings",
"Fishing villages",
"Hot springs",
"Classic board games",
"Desert landscapes",
"Ancient temples",
"Flea markets",
"Old-fashioned diners",
"Medieval castles",
"Wildflower meadows",
"Jazz festivals",
"Secluded beaches",
"Street art alleys",
"Tree-lined boulevards",
"Rural farmhouses",
"Mountain lakes",
"Historic lighthouses",
"Vintage clothing shops",
"Old European town squares",
"Craft breweries",
"Bird watching",
"Ancient libraries",
"Seaside cliffs",
"Classic bookstores",
"Wine cellars",
"Stone cottages",
"Old-fashioned candy shops",
"Historical reenactments",
"Urban gardens",
"Rooftop bars",
"Mountain villages"
]

class CustomParser(Parser):
    """
    The CustomParser class handles the dataset parsing.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self, dataset_path: str, prompt_scheme_path: str, list: List[str]) -> None:
        """
        Initialize the parser.

        :param dataset_path: The path to the dataset.
        :type dataset_path: str
        :param prompt_scheme_path: The path to the prompt scheme file.
        :type prompt_scheme_path: str
        :param list: List of topics to be used in the prompts.
        :type list: List[str]
        """
        super().__init__(dataset_path)
        self.prompt_scheme_path = prompt_scheme_path
        self.list = list

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

        prompt_initial = prompt_complete[0:prompt_complete.find('### HERE ###')]
        prompt_final = prompt_complete[prompt_complete.find('### HERE ###')+len('### HERE ###'):]

        # Use the input data as context inside the prompts
        prompts = []
        for item in self.list:
            prompts.append(prompt_initial + item + prompt_final)

        return prompts

    def ground_truth_extraction(self, custom_inputs: Any = None) -> List[str]:
        """
        Parse the dataset and extract the ground truth.

        :param custom_inputs: The custom inputs to the parser. Defaults to None.
        :type custom_inputs: Any
        :return: List of ground truths.
        :rtype: List[str]
        """
        # Getting the ground truth data from the dataset
        ground_truth = []
        with open(self.dataset_path) as f:
            json_data = json.load(f)

        data_array = json_data['data'][1:]
        for data in data_array:
            text = ""
            for definition in data['definitions']:
                text += definition["term"] + ". " + definition["context"] + "\n"

            text = text[:-1]
            ground_truth.append(text)
        
        return ground_truth

    def answer_parser(self, responses: List[List[str]], custom_inputs: Any = None) -> Any:
        """
        Parse the responses from the model.

        :param responses: The responses from the model.
        :type responses: List[List[str]]
        :param custom_inputs: The custom inputs to the parser. Defaults to None.
        :type custom_inputs: Any
        :return: The parsed responses.
        :rtype: Any
        """
        new_responses = []
        for response in responses:
            new_response = []
            index = response[0].find("### DESCRIPTION 2 ###")
            new_response.append(response[0][:index])
            new_response.append(response[0][index:])
            new_responses.append(new_response)
        return new_responses

def start(current_dir: str, list: List[str]):
    """
    Execute the similar description use case.

    :param current_dir: Directory path from the the script is called.
    :type current_dir: str
    :param list: List of topics to be used in the prompts.
    :type list: List[str]
    """

    config_path = os.path.join(
        current_dir,
        "../../../../CheckEmbed/config.json",
    )

    # Initialize the parser and the embedder
    customParser = CustomParser(
        dataset_path = current_dir,
        prompt_scheme_path = os.path.join(current_dir, "../prompt_scheme.txt"),
        list = list,
    )

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
        budget = 12,
        parser = customParser,
        lm = [gpt4_o, gpt4, gpt3],
        embedding_lm = [embedd_large, sfrEmbeddingMistral, e5mistral7b, gteQwen157bInstruct],
        operations = [bertPlot, selfCheckGPTPlot, rawEmbeddingHeatPlot, checkEmbedPlot],
    )

    # The order of lm_names and embedding_lm_names should be the same 
    # as the order of the language models and embedding language models respectively.
    scheduler.run(
        startingPoint = StartingPoint.PROMPT,
        bertScore = True,
        selfCheckGPT = True,
        ground_truth = False,
        spacy_separator = True,
        num_samples = 1,
        lm_names = ["gpt4-o", "gpt4-turbo", "gpt"],
        embedding_lm_names = ["gpt-embedding-large", "sfr-embedding-mistral", "e5-mistral-7b-instruct", "gte-Qwen15-7B-instruct"],
        bertScore_model = "microsoft/deberta-xlarge-mnli",
        batch_size = 64, # it may be necessary to reduce the batch size if the model is too large
        device = "cuda" # or "cpu" "mps" ...
    )

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/precise"
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    start(current_dir, precise_topics)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/generic"
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    start(current_dir, general_topics)
