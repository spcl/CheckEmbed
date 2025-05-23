# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import json
import logging
import os
import sys
from typing import Any, List

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

from langchain.prompts import PromptTemplate

from CheckEmbed.operations import CheckEmbedOperation, LLMAsAJudgeOperation
from CheckEmbed import language_models
from CheckEmbed import embedding_models
from CheckEmbed.parser import Parser
from CheckEmbed.scheduler import Scheduler, StartingPoint

import CheckEmbedOperation_Variant, \
    BertScoreOperation_Variant, SelfCheckGPT_BERT_Operation_Variant, SelfCheckGPT_NLI_Operation_Variant

prompt_template = PromptTemplate(
    input_variables=["aaa", "bbb"],
    template="""
### INSTRUCTION ###

You are a linguistic and law expert. You will be given a summary of part of a legal document and the original as well. Your job is to rate how accurate and not hallucinated the summary is based on the content of the original passage. You will need to output a score from 0 to 100, where 0 means the summary is completely hallucinated, and 100 means the summary is completely accurate.

### OUTPUT ###

The output should be a single number, which is the score from 0 to 100.
You CANNOT output any other text. You CANNOT output a decimal number. You MUST output an integer number. You MUST NOT output a number that is less than 0 or greater than 100.

### INPUT ###
{aaa}

### ORIGINAL PASSAGE ###
{bbb}
""",
)

class CustomParser(Parser):
    """
    The CustomParser class handles the dataset parsing.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self, dataset_path: str, prompt_scheme_path: str, error_number: int, final_responses_path: str) -> None:
        """
        Initialize the parser.

        :param dataset_path: The path to the dataset.
        :type dataset_path: str
        :param prompt_scheme_path: The path to the prompt scheme file.
        :type prompt_scheme_path: str
        :param error_number: Number of errors that the LLM is asked in the prompts to generate.
        :type error_number: int
        :param final_responses_path: The path to the hallucinations.json file.
        :type final_responses_path: str
        """
        super().__init__(dataset_path)
        self.prompt_scheme_path = prompt_scheme_path
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
        # Getting the input data from the dataset
        input_data = []
        with open(self.dataset_path) as f:
            json_data = json.load(f)

        data_array = json_data['data']
        for data in data_array:
            input_data.append(data['chunk_txt'])

        # Getting the prompt scheme
        with open(self.prompt_scheme_path) as f:
            prompt_complete = f.read()

        # Use input data as context inside the prompts
        prompts = []
        for data in input_data:
            prompt_copy = prompt_complete
            prompts.append(prompt_copy.replace("[###REPLACE WITH CONTEXT###]", data).replace("### NUMBER ###", str(self.error_number)))

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
        :return: The parsed responses. The output is going to be saved in a .json file using the following structure:
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
                index = res.find("### SUMMARY ###")
                new_response.append(res[index:])
                hallucination.append(res[:index])
            new_responses.append(new_response)
            hallucinations.append(hallucination)
        
        with open(self.final_responses_path + "/hallucinations.json", "w") as f:
            hallucinations_json = [{"index": i, "hallucination": hallucinations[i]} for i in range(len(hallucinations))]
            json.dump({"data": hallucinations_json}, f, indent=4)
        return new_responses

def start(current_dir: str, ground_truth_gen: bool = False, error_number: int = 0, start: int = StartingPoint.PROMPT) -> None:
    """
    Execute the incremental forced hallucination use case with a specific error number.

    :param current_dir: Directory path from the the script is called.
    :type current_dir: str
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
        dataset_path = os.path.join(current_dir, "../dataset/legal_definitions.json"),
        prompt_scheme_path = os.path.join(current_dir, "../prompt_scheme.txt" if not ground_truth_gen else "../prompt_scheme_ground_truth.txt"),
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

    gpt4_o_2 = language_models.ChatGPT(
        config_path,
        model_name = "chatgpt4-o",
        cache = True,
        temperature = 0.1,
    )

    gpt4_o_mini = language_models.ChatGPT(
        config_path,
        model_name = "chatgpt4-o-mini",
        cache = False,
        temperature = 0.1,
    )

    llama70 = language_models.LLMChatOllama(
        config_path,
        model_name = "llama70",
        cache = False,
        temperature = 0.1,
    )

    llama8 = language_models.LLMChatOllama(
        config_path,
        model_name = "llama8",
        cache = False,
        temperature = 0.1,
    )

    embedd_large = embedding_models.EmbeddingGPT(
        config_path,
        model_name = "gpt-embedding-large",
        cache = False,
    )

    sfrEmbeddingMistral = embedding_models.SFREmbeddingMistral(
        model_name = "Salesforce/SFR-Embedding-Mistral",
        cache = False,
        batch_size = 8,
    )

    e5mistral7b = embedding_models.E5Mistral7b(
        model_name = "intfloat/e5-mistral-7b-instruct",
        cache = False,
        batch_size = 8,
    )

    gteQwen157bInstruct = embedding_models.GteQwenInstruct(
        model_name= "Alibaba-NLP/gte-Qwen1.5-7B-instruct",
        cache = False,
        access_token = "", # Add your access token here
        batch_size = 4, # it may be necessary to reduce the batch size if the model is too large
    )

    stella_en_15B_v5 = embedding_models.Stella(
        model_name = "NovaSearch/stella_en_1.5B_v5",
        variant = "1.5B-v5",
        cache = False,
    )

    stella_en_400M_v5 = embedding_models.Stella(
        model_name = "NovaSearch/stella_en_400M_v5",
        cache = False,
    )

    llm_judge_Operation = LLMAsAJudgeOperation(
        os.path.join(current_dir, "Judge"),
        current_dir,
        prompt_template = prompt_template,
        original = os.path.join(current_dir, "../../dataset/judge_original.json"),
        original_position = 1,
    )

    # Initialize BERTScore, SelfCheckGPT and CheckEmbedOperation operations
    bertOperation = None if ground_truth_gen else BertScoreOperation_Variant(
        os.path.join(current_dir, "BertScore"),
        os.path.join(current_dir, "../ground_truth"),
        current_dir,
    )

    selfCheckGPTOperation = [] if ground_truth_gen else [
        SelfCheckGPT_BERT_Operation_Variant(
            os.path.join(current_dir, "SelfCheckGPT"),
            os.path.join(current_dir, "../ground_truth"),
            current_dir,
        ),
        SelfCheckGPT_NLI_Operation_Variant(
            os.path.join(current_dir, "SelfCheckGPT"),
            os.path.join(current_dir, "../ground_truth"),
            current_dir,
        )
    ]
    
    checkEmbedOperation = CheckEmbedOperation(
        os.path.join(current_dir, "CheckEmbed_Self"),
        os.path.join(current_dir, "embeddings")
    ) if ground_truth_gen else CheckEmbedOperation_Variant(
            os.path.join(current_dir, "CheckEmbed"),
            os.path.join(current_dir, "../ground_truth/embeddings"),
            os.path.join(current_dir, "embeddings"),
        )

    # Initialize the scheduler
    scheduler = Scheduler(
        current_dir,
        logging_level = logging.DEBUG,
        budget = 10,
        parser = customParser,
        lm = [gpt4_o, gpt3],
        embedding_lm = [stella_en_15B_v5, stella_en_400M_v5, gteQwen157bInstruct, e5mistral7b, sfrEmbeddingMistral, embedd_large],
        bertScoreOperation = bertOperation,
        selfCheckGPTOperation = selfCheckGPTOperation,
        checkEmbedOperation = checkEmbedOperation,
        llm_as_a_judge_Operation = llm_judge_Operation,
        llm_as_a_judge_models = [gpt4_o_mini, gpt4_o_2, llama70, llama8],
    )

    # The order of lm_names and embedding_lm_names should be the same 
    # as the order of the language models and embedding language models respectively.
    scheduler.run(
        startingPoint = start,
        bertScore = True,
        selfCheckGPT = True,
        llm_as_a_judge= True,
        ground_truth = ground_truth_gen,
        spacy_separator = True,
        rebase_results=True,
        num_samples = 10,
        bertScore_model = "microsoft/deberta-xlarge-mnli",
        batch_size = 64, # it may be necessary to reduce the batch size if the model is too large
        device = "cuda" # or "cpu" "mps" ...
    )

if __name__ == "__main__":
    with open(os.path.dirname(os.path.abspath(__file__)) + "/dataset/legal_definitions.json", "r") as f:
        data = json.load(f)["data"]

    with open(os.path.dirname(os.path.abspath(__file__)) + "/dataset/judge_original.json", "w") as f:
        json.dump({"data": [d["chunk_txt"] for d in data]}, f, indent=4)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/ground_truth"
    os.makedirs(current_dir, exist_ok=True)
    os.makedirs(current_dir + "/CheckEmbed_Self", exist_ok=True)
    start(current_dir, ground_truth_gen=True, error_number=0)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_1"
    os.makedirs(current_dir, exist_ok=True)
    start(current_dir, ground_truth_gen=False, error_number=1)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_2"
    os.makedirs(current_dir, exist_ok=True)
    start(current_dir, ground_truth_gen=False, error_number=2)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_3"
    os.makedirs(current_dir, exist_ok=True)
    start(current_dir, ground_truth_gen=False, error_number=3)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_4"
    os.makedirs(current_dir, exist_ok=True)
    start(current_dir, ground_truth_gen=False, error_number=4)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_5"
    os.makedirs(current_dir, exist_ok=True)
    start(current_dir, ground_truth_gen=False, error_number=5)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_6"
    os.makedirs(current_dir, exist_ok=True)
    start(current_dir, ground_truth_gen=False, error_number=6)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_7"
    os.makedirs(current_dir, exist_ok=True)
    start(current_dir, ground_truth_gen=False, error_number=7)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_8"
    os.makedirs(current_dir, exist_ok=True)
    start(current_dir, ground_truth_gen=False, error_number=8)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_9"
    os.makedirs(current_dir, exist_ok=True)
    start(current_dir, ground_truth_gen=False, error_number=9)

    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/error_10"
    os.makedirs(current_dir, exist_ok=True)
    start(current_dir, ground_truth_gen=False, error_number=10)
