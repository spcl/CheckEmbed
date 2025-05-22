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
        input_variables=["aaa", "bbbb"],
        template="""
### INSTRUCTION ###

You are an expert higly skilled in Summarization, Question Answering and in Coverting Data to Text. You will be given an answer to some kind of task, you will also given the original request made by the task. Your task is to evaluate the answer based on the original request, is the answer correct? Or is it factually incorrect and hallucinated?. You will give a score from 0 to 100, where 0 means the answer is completely hallucinated and 100 means the answer is completely correct. Use the full range of scores, 0, 1, 2, ... 10, 20, ... 90, 100.

### OUTPUT ###

The output should be a single number, which is the score from 0 to 100.
You CANNOT output any other text.
You CANNOT output a decimal number.
You MUST output an integer number.
You MUST NOT output a number that is less than 0 or greater than 100.

### ANSWER ###
{aaa}

### ORIGINAL REQUEST ###
{bbbb}
""",
    )

def start(current_dir: str, start: int = StartingPoint.PROMPT, best: bool = False) -> None:
    """
    Start the main function.

    :param current_dir: The current directory.
    :type current_dir: str
    :param num_chunks: The number of chunks. Defaults to 1.
    :type num_chunks: int
    :param start: The starting point. Defaults to StartingPoint.PROMPT.
    :type start: StartingPoint
    """

    # Config file for the LLM(s)
    config_path = os.path.join(
            current_dir,
            "../../CheckEmbed/config.json",
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
        original="../Originals",
        original_position=1,
    )

    embedd_large = embedding_models.EmbeddingGPT(
        config_path=config_path,
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
        batch_size=4, # it may be necessary to reduce the batch size if the GPU VRAM < 40GB
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
        selfCheckGPTOperation=[selfCheckGPT_BERT_Operation, selfCheckGPT_NLI_Operation],
        embedding_lm = [embedd_large, sfrEmbeddingMistral, e5mistral7b, gteQwen157bInstruct, stella_en_400M_v5, stella_en_15B_v5],
        llm_as_a_judge_Operation=llm_judge_Operation,
        llm_as_a_judge_models = [gpt4_o_mini, gpt4_o, llama70, llama8],
    )

    # The order of lm_names and embedding_lm_names should be the same
    # as the order of the language models and embedding language models respectively.
    scheduler.run(
        startingPoint = start,
        selfCheckGPT = best,
        bertScore = best,
        llm_as_a_judge= best,
        checkEmbed = not best,
        rebase_results = True,
        lm_names = ["qa", "summary", "data2text"],
        embedding_lm_names = ["gpt-embedding-large", "sfr-embedding-mistral", "e5-mistral-7B-instruct", "gte-qwen1.5-7B-instruct", "stella-en-400M-v5", "stella-en-1.5B-v5"],
        bertScore_model = "microsoft/deberta-xlarge-mnli",
        batch_size = 64, # it may be necessary to reduce the batch size if the model is too large
        device = "cuda" # or "cpu" "mps" ...
    )

def prepare_data(current_dir: str) -> None:
    if not os.path.exists(current_dir + "/SCGPT_samples"):
        os.makedirs(current_dir + "/SCGPT_samples")

    if not os.path.exists(current_dir + "/Originals"):
        os.makedirs(current_dir + "/Originals")

    data = []
    with open(os.path.join(current_dir, "dataset", "samples.json")) as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))

    with open(os.path.join(current_dir, "dataset", "response.json")) as f:
        data_scgpt = json.load(f)

    with open(os.path.join(current_dir, "dataset", "source_info.json")) as f:
        sources = json.load(f)

    sources_dict = {}
    for source in sources:
        sources_dict[source["source_id"]] = {"task": source["task_type"], "prompt": source["prompt"]}

    data = sorted(data, key=lambda x: x["id"])
    data_scgpt = sorted(data_scgpt, key=lambda x: x["id"])
        
    ce = {
        "QA": [],
        "Summary": [],
        "Data2txt": [],
    }
    scgpt = {
        "QA": [],
        "Summary": [],
        "Data2txt": [],
    }
    original = {
        "QA": [],
        "Summary": [],
        "Data2txt": [],
    }
    for i, elements in enumerate(zip(data, data_scgpt)):
        element = elements[0]
        element_scgpt = elements[1]
        task_type = sources_dict[element["source_id"]]["task"]
        prompt = sources_dict[element["source_id"]]["prompt"]
        original[task_type].append(prompt)
        ce[task_type].append({
            "index": i,
            "samples": [value.strip() for value in element["result"]]
        })
        temp = []
        temp.append(element_scgpt["response"])
        temp.extend([value.strip() for value in element["result"]])
        scgpt[task_type].append({
            "index": i,
            "samples": temp,
        })
    
    with open(os.path.join(current_dir, "SCGPT_samples", "qa_samples.json"), "w") as f:
        json.dump({"data": scgpt["QA"]}, f, indent=4)

    with open(os.path.join(current_dir, "SCGPT_samples", "summary_samples.json"), "w") as f:
        json.dump({"data": scgpt["Summary"]}, f, indent=4)
    
    with open(os.path.join(current_dir, "SCGPT_samples", "data2text_samples.json"), "w") as f:
        json.dump({"data": scgpt["Data2txt"]}, f, indent=4)
    
    with open(os.path.join(current_dir, "qa_samples.json"), "w") as f:
        json.dump({"data": ce["QA"]}, f, indent=4)

    with open(os.path.join(current_dir, "summary_samples.json"), "w") as f:
        json.dump({"data": ce["Summary"]}, f, indent=4)
    
    with open(os.path.join(current_dir, "data2text_samples.json"), "w") as f:
        json.dump({"data": ce["Data2txt"]}, f, indent=4)

    with open(os.path.join(current_dir, "Originals", "qa_original.json"), "w") as f:
        json.dump({"data": original["QA"]}, f, indent=4)
    
    with open(os.path.join(current_dir, "Originals", "summary_original.json"), "w") as f:
        json.dump({"data": original["Summary"]}, f, indent=4)
    
    with open(os.path.join(current_dir, "Originals", "data2text_original.json"), "w") as f:
        json.dump({"data": original["Data2txt"]}, f, indent=4)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    
    prepare_data(current_dir)
    start(current_dir, start=StartingPoint.EMBEDDINGS, best=False)
    start(current_dir, start=StartingPoint.OPERATIONS, best=True)