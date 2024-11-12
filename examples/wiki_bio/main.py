# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import logging
import json
import os

from CheckEmbed import embedding_models
from CheckEmbed.scheduler import Scheduler, StartingPoint
from CheckEmbed.operations import SelfCheckGPT_BERT_Operation, SelfCheckGPT_NLI_Operation

def start(current_dir: str, start: int = StartingPoint.PROMPT, not_ce: bool = False) -> None:
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
            "../../../../../CheckEmbed/config.json",
        )
    
    selfCheckGPT_BERT_Operation = SelfCheckGPT_BERT_Operation(
        os.path.join(current_dir, "SelfCheckGPT"),
        os.path.join(current_dir, "SCGPT_samples"),
    )

    selfCheckGPT_NLI_Operation = SelfCheckGPT_NLI_Operation(
        os.path.join(current_dir, "SelfCheckGPT"),
        os.path.join(current_dir, "SCGPT_samples"),
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

    stella_en_15B_v5 = embedding_models.Stella(
        config_path=config_path,
        model_name = "dunzhang/stella_en_1.5B_v5",
        cache = False,
    )

    stella_en_400M_v5 = embedding_models.Stella(
        config_path=config_path,
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
    )

    # The order of lm_names and embedding_lm_names should be the same
    # as the order of the language models and embedding language models respectively.
    scheduler.run(
        startingPoint = start,
        selfCheckGPT = not_ce,
        checkEmbed = not not_ce,
        bertScore = not_ce,
        rebase_results=True,
        reference_text=True,
        lm_names = ["wikibio"], # Override the language model names
        bertScore_model = "microsoft/deberta-xlarge-mnli",
        batch_size = 64, # it may be necessary to reduce the batch size if the model is too large
        device = "cuda" # or "cpu" "mps" ...
    )

def prepare_data(current_dir: str):
    original_current_dir = current_dir

    for i in range(2, 22, 2):
        current_dir = original_current_dir + f"/{i}_samples"
        os.makedirs(current_dir, exist_ok=True)

        os.makedirs(current_dir + "/embeddings", exist_ok=True)
        
        if i == 20:
            os.makedirs(current_dir + "/SCGPT_samples", exist_ok=True)

            with open(os.path.join(current_dir, "../..", "data", "dataset.json")) as f:
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

def move_embeddings(current_dir: str):
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
    current_dir = os.path.dirname(os.path.abspath(__file__)) + "/results"
    os.makedirs(current_dir, exist_ok=True)
    
    prepare_data(current_dir)

    start(current_dir + "/20_samples", start=StartingPoint.EMBEDDINGS, not_ce = False)
    move_embeddings(current_dir)

    for i in range(2, 22, 2):
        start(current_dir + f"/{i}_samples", start=StartingPoint.OPERATIONS, not_ce = False)

        # The following line is really slow, it is recommended only to get results for i = 20
        start(current_dir + f"/{i}_samples", start=StartingPoint.EMBEDDINGS, not_ce=True)


