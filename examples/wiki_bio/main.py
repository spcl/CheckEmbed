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

from langchain.prompts import PromptTemplate

from CheckEmbed import embedding_models, language_models
from CheckEmbed.operations import (
    LLMAsAJudgeOperation,
    SelfCheckGPT_BERT_Operation,
    SelfCheckGPT_NLI_Operation,
)
from CheckEmbed.scheduler import Scheduler, StartingPoint

prompt_template = PromptTemplate(
    input_variables=["passage"],
    template="""
### INSTRUCTION ###

You are a linguistic and historian expert. You will be given a passage of text containing a brief biography of a famous person. Your job is to rate how hallucinated the biography is. You will need to output a score from 0 to 100, where 0 means the biography is completely hallucinated, and 100 means the biography is completely correct. Use the full range of scores, 0, 1, 2, ... 10, 20, ... 90, 100.

### OUTPUT ###

The output should be a single number, which is the score from 0 to 100.
You CANNOT output any other text.
You CANNOT output a decimal number.
You MUST output an integer number.
You MUST NOT output a number that is less than 0 or greater than 100.

### INPUT ###
{passage}
""",
)

prompt_template_with_ref = PromptTemplate(
    input_variables=["aaa", "bbb"],
    template="""
### INSTRUCTION ###

You are a linguistic and historian expert. You will be given a passage of text containing a brief biography of a famous person, you will also be given the complete original biography of the same famous person. Your job is to rate how hallucinated the biography is compared to the original, longer, one. You will need to output a score from 0 to 100, where 0 means the biography is completely hallucinated, and 100 means the biography is completely correct. Use the full range of scores, 0, 1, 2, ... 10, 20, ... 90, 100. The original biography is always longer, this should be not taken into account as hallucination.

### OUTPUT ###

The output should be a single number, which is the score from 0 to 100.
You CANNOT output any other text.
You CANNOT output a decimal number.
You MUST output an integer number.
You MUST NOT output a number that is less than 0 or greater than 100.

### INPUT ###
**Passage**:
{aaa}

**Original**:
{bbb}
""",
)

def start(current_dir: str, start: int = StartingPoint.PROMPT, not_ce: bool = False, llm_as_a_judge: bool = False, llm_as_a_judge_with_ref: bool = False) -> None:
    """
    Start the main function.

    :param current_dir: The current directory.
    :type current_dir: str
    :param start: The starting point. Defaults to StartingPoint.PROMPT.
    :type start: StartingPoint
    :param not_ce: Flag to indicate whether we execute the CheckEmbed operation. Defaults to False.
    :type not_ce: bool
    """

    # Config file for the LLM(s)
    config_path = os.path.join(
            current_dir,
            "../../../CheckEmbed/config.json",
        )
    
    if llm_as_a_judge or llm_as_a_judge_with_ref:
        # Config file for the LLM(s)
        config_path = os.path.join(
            current_dir,
            "../../CheckEmbed/config.json",
        )
    
    selfCheckGPT_BERT_Operation = SelfCheckGPT_BERT_Operation(
        os.path.join(current_dir, "SelfCheckGPT"),
        os.path.join(current_dir, "SCGPT_samples"),
    )

    selfCheckGPT_NLI_Operation = SelfCheckGPT_NLI_Operation(
        os.path.join(current_dir, "SelfCheckGPT"),
        os.path.join(current_dir, "SCGPT_samples"),
    )

    llm_judge_Operation = LLMAsAJudgeOperation(
        os.path.join(current_dir, "Judge"),
        current_dir,
        prompt_template = prompt_template,
    )

    llm_judge_Operation_with_ref = LLMAsAJudgeOperation(
        os.path.join(current_dir, "Judge"),
        current_dir,
        prompt_template = prompt_template,
        original = os.path.join(current_dir, "../original_samples.json"),
        original_position = 1,
        reference_txt = "ref",
    )

    gpt4_o = language_models.ChatGPT(
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
    )

    e5mistral7b = embedding_models.E5Mistral7b(
        model_name = "intfloat/e5-mistral-7b-instruct",
        cache = False,
    )

    gteQwen157bInstruct = embedding_models.GteQwenInstruct(
        model_name = "Alibaba-NLP/gte-Qwen1.5-7B-instruct",
        cache = False,
        access_token = "", # Add your access token here
        batch_size = 4, # it may be necessary to reduce the batch size if the GPU VRAM < 40GB
    )

    stella_en_15B_v5 = embedding_models.Stella(
        model_name = "dunzhang/stella_en_1.5B_v5",
        variant = "1.5B-v5",
        cache = False,
    )

    stella_en_400M_v5 = embedding_models.Stella(
        model_name = "dunzhang/stella_en_400M_v5",
        cache = False,
    )

    # Initialize the scheduler
    scheduler = Scheduler(
        current_dir,
        logging_level = logging.DEBUG,
        budget = 30,
        selfCheckGPTOperation = [selfCheckGPT_BERT_Operation, selfCheckGPT_NLI_Operation],
        embedding_lm = [embedd_large, sfrEmbeddingMistral, e5mistral7b, gteQwen157bInstruct, stella_en_400M_v5, stella_en_15B_v5],
        llm_as_a_judge_Operation=llm_judge_Operation if llm_as_a_judge else llm_judge_Operation_with_ref,
        llm_as_a_judge_models = [gpt4_o_mini, gpt4_o, llama70, llama8],
    )

    # The order of lm_names and embedding_lm_names should be the same
    # as the order of the language models and embedding language models respectively.
    scheduler.run(
        startingPoint = start,
        selfCheckGPT = not_ce and not llm_as_a_judge and not llm_as_a_judge_with_ref,
        llm_as_a_judge = llm_as_a_judge or llm_as_a_judge_with_ref,
        checkEmbed = not not_ce and not llm_as_a_judge and not llm_as_a_judge_with_ref,
        bertScore = not_ce and not llm_as_a_judge and not llm_as_a_judge_with_ref,
        rebase_results = True,
        reference_text = True,
        lm_names = ["wikibio"], # Override the language model names
        bertScore_model = "microsoft/deberta-xlarge-mnli",
        batch_size = 64, # it may be necessary to reduce the batch size if the model is too large
        device = "cuda" # or "cpu" "mps" ...
    )

def prepare_data(current_dir: str) -> None:
    """
    Prepare the data.

    :param current_dir: Directory, from where the script is executed.
    :type current_dir: str
    """
    original_current_dir = current_dir

    for i in range(2, 22, 2):
        current_dir = original_current_dir + f"/{i}_samples"
        os.makedirs(current_dir, exist_ok=True)

        os.makedirs(current_dir + "/embeddings", exist_ok=True)
        
        if i == 20:
            os.makedirs(current_dir + "/SCGPT_samples", exist_ok=True)

            with open(os.path.join(current_dir, "..", "data", "dataset.json")) as f:
                data: dict = json.load(f)

            scgpt = []
            ce = []

            for key, value in zip(data.keys(), data.values()):
                ce.append({
                    "index": int(key.replace("passage_", "")),
                    "samples": value["gpt3_text_samples"]
                })
                temp = []
                temp.append(value["gpt3_sentences"])
                temp.extend(value["gpt3_text_samples"])
                scgpt.append({
                    "index": int(key.replace("passage_", "")),
                    "samples": temp
                })

            with open(current_dir + "/wikibio_samples.json", "w") as f:
                json.dump({"data" : ce}, f, indent=4)
            
            with open(current_dir + "/SCGPT_samples/wikibio_samples.json", "w") as f:
                json.dump({"data" : scgpt}, f, indent=4)

    with open(os.path.join(original_current_dir, "data", "dataset.json")) as f:
        data: dict = json.load(f)

    sample = [data[f"passage_{i}"]["gpt3_text"] for i in range(238)]
    original = [data[f"passage_{i}"]["wiki_bio_text"] for i in range(238)]

    sample_json = []
    for s in sample:
        sample_json.append({
            "samples": [s]
        })

    with open(original_current_dir + "/wikibio_samples.json", "w") as f:
        json.dump({"data": sample_json}, f, indent=4)

    with open(original_current_dir + "/original_samples.json", "w") as f:
        json.dump({"data": original}, f, indent=4)

def move_embeddings(current_dir: str) -> None:
    """
    Move the embedding data to a different location.

    :param current_dir: Directory, from where the script is executed.
    :type current_dir: str
    """
    original_current_dir = current_dir
    embedding_dir = current_dir + "/20_samples" + "/embeddings"

    embeddings_dict = {}
    for file in os.listdir(embedding_dir):
        with open(embedding_dir + "/" + file, "r") as f:
            embeddings_dict.update({
                file: json.load(f)
            })
    
    for i in range(2, 20, 2):
        current_dir = original_current_dir + f"/{i}_samples/embeddings"
        for file, value in zip(embeddings_dict.keys(), embeddings_dict.values()):
            new_value = []
            for embedding in value["data"]:
                new_value.append({
                    "prompt_index": embedding["prompt_index"],
                    "embeddings": embedding["embeddings"][:i]
                })
            with open(current_dir + "/" + file, "w") as f:
                json.dump({"data": new_value}, f, indent=4)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(current_dir, exist_ok=True)
    
    prepare_data(current_dir)

    start(current_dir + "/20_samples", start=StartingPoint.EMBEDDINGS, not_ce = False)
    move_embeddings(current_dir)

    for i in range(2, 22, 2):
        start(current_dir + f"/{i}_samples", start=StartingPoint.OPERATIONS, not_ce = False)

        # The following line is really slow, it is recommended only to get results for i = 20
        start(current_dir + f"/{i}_samples", start=StartingPoint.EMBEDDINGS, not_ce = True)

    start(current_dir, start=StartingPoint.OPERATIONS, not_ce = True, llm_as_a_judge = True)
    start(current_dir, start=StartingPoint.OPERATIONS, not_ce = True, llm_as_a_judge_with_ref = True)
