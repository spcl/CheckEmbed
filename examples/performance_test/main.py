# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import logging
import os
import json
import random
import tiktoken

from faker import Faker
from datetime import datetime as time

from CheckEmbed import embedding_models
from CheckEmbed.scheduler import Scheduler, StartingPoint
from CheckEmbed.operations import SelfCheckGPT_BERT_Operation, SelfCheckGPT_NLI_Operation

def start(current_dir: str, start: int = StartingPoint.PROMPT, n_samples: int = 10) -> None:
    """
    Execute the runtime measurements.

    :param current_dir: Directory path from the the script is called.
    :type current_dir: str
    :param start: The starting point of the scheduler. Defaults to StartingPoint.PROMPT.
    :type start: int
    :param n_samples: Number of samples to generate. Defaults to 10.
    :type n_samples: int
    """

    config_path = os.path.join(
        current_dir,
        "../../../CheckEmbed/config.json",
    )

    embedd_large = embedding_models.EmbeddingGPT(
        config_path,
        model_name = "gpt-embedding-large",
        cache = False,
        max_concurrent_requests=5,
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
        config_path = config_path,
        model_name = "Alibaba-NLP/gte-Qwen1.5-7B-instruct",
        cache = False,
        access_token = "", # Add your access token here (Hugging Face)
    )

    stella_en_15B_v5 = embedding_models.Stella(
        config_path = config_path,
        model_name = "dunzhang/stella_en_1.5B_v5",
        variant = "1.5B-v5",
        cache = False,
    )

    stella_en_400M_v5 = embedding_models.Stella(
        config_path = config_path,
        model_name = "dunzhang/stella_en_400M_v5",
        cache = False,
    )

    selfCheckGPT_BERT_Operation = SelfCheckGPT_BERT_Operation(
        os.path.join(current_dir, "SelfCheckGPT"),
        current_dir,
    )

    selfCheckGPT_NLI_Operation = SelfCheckGPT_NLI_Operation(
        os.path.join(current_dir, "SelfCheckGPT"),
        current_dir,
    )

    # Initialize the scheduler
    scheduler = Scheduler(
        current_dir,
        logging_level = logging.DEBUG,
        budget = 8,
        selfCheckGPTOperation=[selfCheckGPT_BERT_Operation, selfCheckGPT_NLI_Operation],
        embedding_lm = [embedd_large, sfrEmbeddingMistral, e5mistral7b, gteQwen157bInstruct, stella_en_400M_v5, stella_en_15B_v5],
    )

    # The order of lm_names and embedding_lm_names should be the same 
    # as the order of the language models and embedding language models respectively.
    scheduler.run(
        startingPoint = start,
        bertScore = True, # Set to True if you want to test BERTScore
        selfCheckGPT = True, # Set to True if you want to test SelfCheckGPT
        time_performance = True,
        num_samples = n_samples,
        lm_names = [str(i) for i in range(200, 4200, 200)], # Overwrite the default lm names
        bertScore_model = "microsoft/deberta-xlarge-mnli",
        device = "cuda",
        batch_size = 64 # it may be necessary to reduce the batch size if the model is too large
    )


def text_gen(n_prompt: int = 50, n_samples: int = 10, dir: str = ".") -> None:
    """
    Generate text with different number of tokens for a specific number of samples.

    :param n_prompt: Number of datapoints for a specific combination of token size and number of
                     samples. Defaults to 50.
    :type n_prompt: int
    :param n_samples: Number of samples. Default to 10.
    :type n_samples: int
    :param dir: Path to the output directory. Defaults to the current directory.
    :type dir: str
    """

    fake = Faker()
    fake.seed_instance(int(random.Random(time.now().microsecond).random() * 1000))

    fake.name()
    fake.address()

    encoding = tiktoken.get_encoding("cl100k_base")
    
    for length in range(200, 4200, 200):
        len_samples = []
        for _ in range(n_prompt):
            samples = []
            for _ in range(n_samples):
                temp = fake.text(max_nb_chars=length*10).replace("\n", " ")
                while len(encoding.encode(temp)) < length:
                    temp += fake.text(max_nb_chars=length*10).replace("\n", " ")

                final_dimension = len(encoding.encode(temp))

                # Add the samples to the list and keep only around the desired token length
                samples.append(temp[0:int(len(temp) * (length / final_dimension))])
            len_samples.append(samples)
        
        with open(f"{dir}/{length}_samples.json", "w") as f:
            json_data = [{"index": i, "samples": samples} for i, samples in enumerate(len_samples)]
            json.dump({"data": json_data}, f, indent=4)


if __name__ == "__main__":
    print("Performance test\n")

    for sample_count in [2, 4, 6, 8, 10]:
        print(f"\n\n\n#########################\n#\t{sample_count} SAMPLES\t#\n#########################")
        current_dir = os.path.dirname(os.path.abspath(__file__)) + f"/{sample_count}_samples"
        os.makedirs(current_dir, exist_ok=True)
        text_gen(20, n_samples=sample_count, dir=f"{sample_count}_samples")
        start(current_dir, start=StartingPoint.EMBEDDINGS, n_samples=sample_count)
