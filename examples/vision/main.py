# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Eric Schreiber
# 
# Contributors: Lorenzo Paleari

import logging
import os
from typing import Any, List, Union

from PIL.Image import Image

from CheckEmbed import embedding_models, vision_models
from CheckEmbed.parser import Parser
from CheckEmbed.scheduler import Scheduler, StartingPoint

input_prompts = [
    ["One red apple on a white background",
    "Two red apples on a white background",
    "Three red apples on a white background",
    "Four red apples on a white background",
    "Five red apples on a white background",
    ],
    ["One yellow tennis ball on a white background",
    "Two yellow tennis balls on a white background",
    "Three yellow tennis balls on a white background",
    "Four yellow tennis balls on a white background",
    "Five yellow tennis balls on a white background",
    ],
    ["One orange on a white background",
    "Two oranges on a white background",
    "Three oranges on a white background",
    "Four oranges on a white background",
    "Five oranges on a white background",
    ],
    ["One yellow lemon on a white background",
    "Two yellow lemons on a white background",
    "Three yellow lemons on a white background",
    "Four yellow lemons on a white background",
    "Five yellow lemons on a white background",
    ],
    ["One green lime on a white background",
    "Two green limes on a white background",
    "Three green limes on a white background",
    "Four green limes on a white background",
    "Five green limes on a white background",
    ],
    ["One red tomato on a white background",
    "Two red tomatoes on a white background",
    "Three red tomatoes on a white background",
    "Four red tomatoes on a white background",
    "Five red tomatoes on a white background",
    ],
    ["One yellow banana on a white background",
    "Two yellow bananas on a white background",
    "Three yellow bananas on a white background",
    "Four yellow bananas on a white background",
    "Five yellow bananas on a white background",
    ],
    ["One blue circle on a white background",
    "Two blue circles on a white background",
    "Three blue circles on a white background",
    "Four blue circles on a white background",
    "Five blue circles on a white background",
    ]
]

class CustomParser(Parser):
    """
    The CustomParser class handles the dataset parsing.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self, dataset_path: str, list: List[str]) -> None:
        """
        Initialize the parser.

        :param dataset_path: The path to the dataset.
        :type dataset_path: str
        :param list: The list of two different topics to be used in the prompts.
        :type list: List[str]
        """
        super().__init__(dataset_path)
        self.list = list

    def prompt_generation(self, custom_inputs: Any = None) -> List[str]:
        """
        Parse the dataset and generate the prompts for the model.

        :param custom_inputs: The custom inputs to the parser. Defaults to None.
        :type custom_inputs: Any
        :return: List of prompts.
        :rtype: List[str]
        """
        prompts = []
        for item in self.list:
            prompts.extend(item)

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

    def answer_parser(self, responses: List[List[Union[str, Image]]], custom_inputs: Any = None) -> List[List[Union[str, Image]]]:
        """
        Parse the responses from the model.

        The default behavior is to return the responses as they are.
        Overwrite this method if you want to parse the responses in a different way. You can use the CustomParser
        classes in the examples folder as reference.

        Remember that the responses returned from this method will be stored in a file and used for the evaluation,
        so please follow the following format, when returning the responses:
        [
            [response1_prompt1, response2_prompt1, ...],
            [response1_prompt2, response2_prompt2, ...],
            ...
        ]

        :param responses: The responses from the model.
        :type responses: List[List[Union[str, Image]]]
        :param custom_inputs: The custom inputs to the parser. Defaults to None.
        :type custom_inputs: Any
        :return: The parsed responses.
        :rtype: List[List[Union[str, Image]]]
        """
        return responses


def start(current_dir: str, list: List[str]) -> None:
    """
    Execute the different description use case.

    :param current_dir: Directory path from the the script is called.
    :type current_dir: str
    :param list: The list of two different topics to be used in the prompts.
    :type list: List[str]
    """

    # Initialize the parser and the embedder
    customParser = CustomParser(
        dataset_path = current_dir,
        list = list
    )

    stable_diffusion = vision_models.StableDiffusion3(
        model_name = "stabilityai/stable-diffusion-3.5-medium",
        cache = False,
    )

    clip_vit_large = embedding_models.ClipVitLarge(
        model_name = "openai/clip-vit-large-patch14",
        cache = False,
    )

    # Initialize the scheduler
    scheduler = Scheduler(
        current_dir,
        logging_level = logging.DEBUG,
        budget = 12,
        parser = customParser,
        lm = [stable_diffusion],
        embedding_lm = [clip_vit_large],
    )

    # The order of lm_names and embedding_lm_names should be the same 
    # as the order of the language models and embedding language models respectively.
    scheduler.run(
        startingPoint = StartingPoint.PROMPT,
        bertScore = False,
        selfCheckGPT = False,
        llm_as_a_judge = False,
        vision = True,
        rebase_results=True,
        num_samples = 10,
        bertScore_model = "microsoft/deberta-xlarge-mnli",
        device = "cuda",
        batch_size = 64 # it may be necessary to reduce the batch size if the model is too large
    )

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    start(current_dir, input_prompts)
