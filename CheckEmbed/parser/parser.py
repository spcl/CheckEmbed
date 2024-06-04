# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

from abc import ABC, abstractmethod

from typing import Any, List

class Parser(ABC):
    """
    Abstract base class that defines the interface for parsing.

    The class supports the following functionality:
    - take the raw data from a dataset and create the necessary prompts for the model
    - extract the ground truth
    - custom parsing of the model responses
    """

    def __init__(self, dataset_path: str) -> None:
        """
        Initialize the parser.

        :param dataset_path: The path to the dataset.
        :type dataset_path: str
        """
        self.dataset_path = dataset_path

    @abstractmethod
    def prompt_generation(self, custom_inputs: Any = None) -> List[str]:
        """
        Parse the dataset and generate the prompts for the model.

        :param custom_inputs: The custom inputs to the parser. Defaults to None.
        :type custom_inputs: Any
        :return: List of prompts.
        :rtype: List[str]
        """
        pass
    
    @abstractmethod
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
        :type responses: List[List[str]]
        :param custom_inputs: The custom inputs to the parser. Defaults to None.
        :type custom_inputs: Any
        :return: The parsed responses.
        :rtype: List[List[str]]
        """
        return responses
