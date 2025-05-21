# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import os
import json

from typing import Any

from CheckEmbed.operations import Operation
from CheckEmbed.language_models import ChatGPT
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

class Score(BaseModel):
    score: int = Field(description="The score from 0 to 100")

class LLMAsAJudgeOperation(Operation):
    """
    Operation that computes the hallucination score of an answer using a language model as a judge.

    Inherits from the Operation class and implements its abstract methods.
    """

    def __init__(self, result_dir_path: str, answer_dir_path: str, prompt_template: PromptTemplate, original: str = None, original_position: int = 0, reference_txt: str = None) -> None:
        """
        Initialize the operation.

        :param result_dir_path: The path to the directory where the results will be stored.
        :type result_dir_path: str
        :param answer_dir_path: The path to the directory where the answers are stored.
        :type answer_dir_path: str
        :param prompt_template: The prompt template to be used for the language model.
        :type prompt_template: PromptTemplate
        :param original: The original data.
        :type original: str
        :param original_position: The position of the original data in the prompt template.
        :type original_position: int
        """
        super().__init__(result_dir_path)
        self.answer_dir_path = answer_dir_path
        self.prompt_template = prompt_template
        self.original = original
        self.original_position = original_position
        self.reference_txt = reference_txt

    def execute(self, custom_inputs: Any) -> Any:
        """
        Execute the operation on the embeddings/samples.

        :param custom_inputs: The custom inputs for the operation.
        :type custom_inputs: Any
        """
        model = custom_inputs["model"]
        if not isinstance(model, ChatGPT):
            model.add_structured_output(Score)

        if self.original is not None:
            with open(self.result_dir_path + self.original, "r") as f:
                original_data = json.load(f)["data"]

        # For every language model
        for file in os.listdir(self.answer_dir_path):
            if "samples.json" in file and not file.startswith("ground_truth_"):
                
                name = model.name + "_" + file.split("_")[0]
                if name.startswith("gpt4-o"):
                    name = name[6:]
                    name = "4o" + name

                # Load the samples
                with open(os.path.join(self.answer_dir_path, file), "r") as f:
                    data = json.load(f)
                data_array = data["data"]
                samples = [d["samples"] for d in data_array]

                inputs = self.prompt_template.input_variables

                results = []
                if self.original is not None:
                    for i, sample in enumerate(samples):
                        prep = {}
                        for j, input in enumerate(inputs):
                            if j == self.original_position:
                                prep[input] = original_data[i]
                            else:
                                prep[input] = sample[j]

                        final_prompt = self.prompt_template.invoke(prep)
                        if isinstance(model, ChatGPT):
                            final_prompt = final_prompt.text
                        result = model.query(final_prompt)
                        if not isinstance(result, Score):
                            result = model.get_response_texts(result)[0]
                        else:
                            result = result.score

                        results.append(result)
                else:
                    for sample in samples:
                        prep = {}
                        for i, input in enumerate(inputs):
                            prep[input] = sample[i]

                        final_prompt = self.prompt_template.invoke(prep)
                        if isinstance(model, ChatGPT):
                            final_prompt = final_prompt.text
                        result = model.query(final_prompt)
                        if not isinstance(result, Score):
                            result = model.get_response_texts(result)[0]
                        else:
                            result = result.score

                        results.append(result)

                # Store the results
                with open(os.path.join(self.result_dir_path, name + "_judge.json"), "w") as f:
                    json.dump({"data": results}, f, indent=4)

                
