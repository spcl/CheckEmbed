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

different_topics_list = [
["sunrise over a mountain range", "a bustling city street"],
["a vintage typewriter", "a futuristic smart home"],
["an ancient oak tree", "a modern art sculpture"],
["a secluded beach", "a crowded amusement park"],
["a traditional Japanese tea ceremony", "a rock concert"],
["a cozy fireplace", "a high-tech office"],
["a medieval castle", "a sleek sports car"],
["a field of wildflowers", "a space station"],
["an old library", "a tropical rainforest"],
["a farmer's market", "a luxury yacht"],
["a quiet village", "a grand cathedral"],
["a serene lake", "a busy airport terminal"],
["a historical battlefield", "a contemporary dance performance"],
["a rustic cabin", "a high-rise apartment building"],
["a botanical garden", "a robotics lab"],
["a snowy mountain peak", "a desert oasis"],
["a blacksmith's forge", "a modern fashion runway"],
["a children's playground", "a gourmet restaurant kitchen"],
["an underwater coral reef", "a bustling train station"],
["a classical music concert hall", "a professional sports stadium"],
["a tropical island", "an urban graffiti wall"],
["a quaint bookstore", "a cutting-edge tech conference"],
["a lavender field", "a metropolitan subway system"],
["a windmill farm", "a crowded beach boardwalk"],
["an antique shop", "a futuristic car showroom"],
["a serene monastery", "a vibrant city festival"],
["a cozy coffee shop", "a high-speed train"],
["a haunted house", "a sunny vineyard"],
["a mountain hiking trail", "a rooftop garden"],
["a historic lighthouse", "a bustling shopping mall"],
["a tranquil zen garden", "a lively street market"],
["a snow-covered village", "a modern art museum"],
["a dense jungle", "a high-tech gaming arcade"],
["a medieval marketplace", "a contemporary office building"],
["a quiet fishing village", "a grand opera house"],
["a colorful coral reef", "a busy urban intersection"],
["a serene temple", "a neon-lit cityscape"],
["a peaceful meadow", "a state-of-the-art hospital"],
["an artist's studio", "a crowded stadium"],
["a small countryside farm", "a luxurious hotel lobby"],
["a moonlit forest", "a vibrant night club"],
["a rustic bakery", "a high-tech research lab"],
["a sunflower field", "a busy construction site"],
["a quaint cottage", "a sprawling industrial complex"],
["a quiet cemetery", "a bustling open-air market"],
["a serene mountain lake", "a chaotic newsroom"],
["a vintage record store", "a modern science museum"],
["a sleepy fishing harbor", "a high-energy concert venue"],
["a peaceful riverbank", "a bustling stock exchange"],
["an old-fashioned diner", "a futuristic spaceport"],
["a medieval cathedral", "a high-rise corporate office"],
["a tranquil waterfall", "a fast-paced city street"],
["a historic museum", "a cutting-edge robotics factory"],
["a cozy reading nook", "a busy public transit hub"],
["a lush vineyard", "a high-tech kitchen"],
["a classic car show", "a modern art gallery"],
["a scenic mountain trail", "a packed football stadium"],
["a serene monastery garden", "a busy airport runway"],
["a quaint village square", "a bustling food market"],
["a peaceful desert landscape", "a neon-lit downtown area"],
["an old stone bridge", "a futuristic skyscraper"],
["a quiet country road", "a busy shopping district"],
["a historical reenactment", "a high-tech startup office"],
["a serene lake house", "a packed music festival"],
["a classic library", "a modern coworking space"],
["a peaceful hilltop view", "a busy urban plaza"],
["an old-fashioned blacksmith shop", "a state-of-the-art gym"],
["a rustic mountain cabin", "a luxurious spa resort"],
["a cozy home kitchen", "a bustling city harbor"],
["a quiet woodland trail", "a vibrant street parade"],
["a tranquil beach at sunset", "a crowded urban park"],
["a vintage photography studio", "a modern film set"],
["a sleepy village inn", "a busy metropolitan police station"],
["a historic battlefield site", "a contemporary music studio"],
["a quaint bed and breakfast", "a high-tech medical center"],
["a scenic coastal road", "a bustling city cafe"],
["a traditional pottery workshop", "a modern tech exhibition"],
["a peaceful village pond", "a crowded airport lounge"],
["a rustic farm barn", "a sleek urban penthouse"],
["a tranquil forest glade", "a vibrant city street fair"],
["a quiet monastery library", "a packed city sports arena"],
["a scenic mountain overlook", "a modern urban art installation"],
["a cozy lakeside cabin", "a bustling urban market"],
["a historic shipyard", "a futuristic transportation hub"],
["a serene countryside meadow", "a busy metropolitan newsroom"],
["a rustic vineyard cellar", "a high-tech digital lab"],
["a peaceful riverside park", "a bustling city square"],
["a quaint rural village", "a modern shopping center"],
["a scenic woodland clearing", "a vibrant city street corner"],
["a historic castle ruin", "a contemporary dance club"],
["a tranquil botanical conservatory", "a crowded urban nightclub"],
["a rustic farmhouse kitchen", "a modern high-rise office"],
["a quiet village green", "a bustling city intersection"],
["a serene mountain meadow", "a lively urban festival"],
["a vintage train station", "a modern airport terminal"],
["a peaceful woodland cabin", "a high-tech corporate headquarters"],
["a quaint countryside church", "a bustling city theater"]
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
        :param list: The list of two different topics to be used in the prompts.
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

        prompt_first = prompt_complete[0:prompt_complete.find('### HERE 1 ###')]
        prompt_second = prompt_complete[prompt_complete.find('### HERE 1 ###')+len('### HERE 1 ###'):prompt_complete.find('### HERE 2 ###')]
        prompt_last = prompt_complete[prompt_complete.find('### HERE 2 ###')+len('### HERE 2 ###'):]

        # Use the input data as context inside the prompts
        prompts = []
        for item in self.list:
            prompts.append(prompt_first + item[0] + prompt_second + item[1] + prompt_last)

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

    def answer_parser(self, responses: List[List[str]], custom_inputs: Any = None) -> List[List[str]]:
        """
        Parse the responses from the model.

        :param responses: The responses from the model.
        :type responses: List[List[str]]
        :param custom_inputs: The custom inputs to the parser. Defaults to None.
        :type custom_inputs: Any
        :return: The parsed responses.
        :rtype: List[List[str]]
        """
        new_responses = []
        for response in responses:
            new_response = []
            index = response[0].find("### DESCRIPTION 2 ###")
            new_response.append(response[0][:index])
            new_response.append(response[0][index:])
            new_responses.append(new_response)
        return new_responses


def start(current_dir: str, list: List[str]) -> None:
    """
    Execute the different description use case.

    :param current_dir: Directory path from the the script is called.
    :type current_dir: str
    :param list: The list of two different topics to be used in the prompts.
    :type list: List[str]
    """

    config_path = os.path.join(
        current_dir,
        "../../../CheckEmbed/config.json",
    )

    # Initialize the parser and the embedder
    customParser = CustomParser(
        dataset_path = current_dir,
        prompt_scheme_path = "prompt_scheme.txt",
        list = list
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
        access_token = "", # Add your access token here (Hugging Face)
    )

    # Initialize the operations
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
        device = "cuda",
        batch_size = 64 # it may be necessary to reduce the batch size if the model is too large
    )

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    start(current_dir, different_topics_list)
