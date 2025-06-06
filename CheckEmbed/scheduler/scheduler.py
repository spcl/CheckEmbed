# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import os
import json
import logging

from enum import Enum
from typing import List, Union
from timeit import default_timer as timer
from tqdm import tqdm
from PIL import Image

from CheckEmbed.language_models import AbstractLanguageModel
from CheckEmbed.embedding_models import AbstractEmbeddingModel
from CheckEmbed.vision_models import AbstractVisionModel
from CheckEmbed.operations import BertScoreOperation
from CheckEmbed.operations import CheckEmbedOperation
from CheckEmbed.operations import SelfCheckGPT_Operation, SelfCheckGPT_NLI_Operation
from CheckEmbed.operations import Operation
from CheckEmbed.operations import LLMAsAJudgeOperation
from CheckEmbed.parser import Parser
from CheckEmbed.embedder import Embedder

class StartingPoint(Enum):
    """
    Enum representing the starting point options for the scheduler.
    
    Attributes:
        PROMPT (int): All operations will be executed in order, starting from the prompt generation.
        SAMPLES (int): All operations will be executed in order starting from the sample generation using the LLM.
        EMBEDDINGS (int): All operations will be executed in order, starting from the embedding generation.
        OPERATIONS (int): All operations will be executed in order: BERTScore, SelfCheckGPT, CheckEmbed and all other operations specified in the scheduler class parameters. To disable any of the first three operations, set the corresponding flag to False in the run method.
    """
    
    PROMPT = 1
    SAMPLES = 2
    EMBEDDINGS = 3
    OPERATIONS = 4

class Scheduler:
    """
    The Scheduler class is responsible for coordinating the execution of various operations in the verification process.
    It manages the generation of prompts, samples, embeddings, and the execution of operations such as BERTScore and SelfCheckGPT as well as other custom ones.
    """

    def __init__(
            self, 
            workdir: str,
            logging_level: int = logging.INFO,
            budget: int = 10,
            parser: Parser = None, 
            embedder: Embedder = Embedder(), 
            lm: List[Union[AbstractLanguageModel, AbstractVisionModel]] = None,
            embedding_lm: List[AbstractEmbeddingModel] = None,
            operations: List[Operation] = [],
            bertScoreOperation: BertScoreOperation = None,
            selfCheckGPTOperation: List[SelfCheckGPT_Operation] = [],
            checkEmbedOperation: CheckEmbedOperation = None,
            llm_as_a_judge_Operation: LLMAsAJudgeOperation = None,
            llm_as_a_judge_models: List[AbstractLanguageModel] = [],
        ) -> None:
        """
        Initializes the Scheduler instance with the given parameters.
        
        :param workdir: The working directory where the generated data and logs will be stored.
        :type workdir: str
        :param logging_level: The logging level for the scheduler and its operations.
        :type logging_level: int
        :param budget: The budget in Dollars for the sampling and embedding process. Defaults to 10 Dollars.
        :type budget: int
        :param parser: An instance of the Parser class required for the prompt and ground truth generation. Defaults to None.
        :type parser: Parser
        :param embedder: An instance of the Embedder class required for the embedding generation. Defaults to the abstract Embedder.
        :type embedder: Embedder
        :param lm: A list of AbstractLanguageModel or AbstractVisionModel instances representing the models used for sampling. Defaults to None.
        :type lm: List[Union[AbstractLanguageModel, AbstractVisionModel]]
        :param embedding_lm: A list of AbstractEmbeddingModel instances representing the embedding models used for the embedding generation. Defaults to None.
        :type embedding_lm: List[AbstractEmbeddingModel]
        :param operations: A list of Operation instances representing additional operations to be executed. Defaults to an empty list.
        :type operations: List[Operation]
        :param bertScoreOperation: An instance of a custom BertScoreOperation class for the BERTScore computation. Defaults to None. If None, the default BertScoreOperation will be used.
        :type bertScoreOperation: BertScoreOperation
        :param selfCheckGPTOperation: A list of instances of a custom selfCheckGPTOperation class for the SelfCheckGPT computation. Defaults to an empty list. If Empty, the default SelfCheckGPT_NLI_Operation will be used.
        :type selfCheckGPTOperation: List[SelfCheckGPT_Operation]
        :param checkEmbedOperation: An instance of a custom CheckEmbedOperation class for CheckEmbed computation. Defaults to None. If None, the default CheckEmbedOperation will be used.
        :type checkEmbedOperation: CheckEmbedOperation
        :param llm_as_a_judge_Operation: An instance of a custom LLMAsAJudgeOperation class for the LLM as a judge computation. Defaults to None.
        :type llm_as_a_judge_Operation: LLMAsAJudgeOperation
        :param llm_as_a_judge_models: A list of AbstractLanguageModel instances representing the language models used for the LLM as a judge operation. Defaults to an empty list.
        :type llm_as_a_judge_models: List[AbstractLanguageModel]
        """

        self.workdir = workdir
        self.logging_level = logging_level
        self.parser = parser
        self.embedder = embedder
        self.lm = lm
        self.embedding_lm = embedding_lm
        self.budget = budget
        self.operations = operations
        self.bertScoreOperation = BertScoreOperation(os.path.join(workdir, "BertScore"), workdir) if bertScoreOperation is None else bertScoreOperation
        self.selfCheckGPTOperation = [SelfCheckGPT_NLI_Operation(os.path.join(workdir, "SelfCheckGPT"), workdir)] if len(selfCheckGPTOperation) == 0 else selfCheckGPTOperation
        self.checkEmbedOperation = CheckEmbedOperation(os.path.join(workdir, "CheckEmbed"), os.path.join(workdir, "embeddings")) if checkEmbedOperation is None else checkEmbedOperation
        self.llm_as_a_judge_Operation = llm_as_a_judge_Operation
        self.llm_as_a_judge_models = llm_as_a_judge_models


    def _prompt_generation(self) -> bool:
        """
        Generate the prompts for the model and save them to a json file.

        :return: False if the parser is missing, True otherwise.
        :rtype: bool
        """

        if self.parser is None:
            return False

        prompts = self.parser.prompt_generation()
        with open(os.path.join(self.workdir, "prompts.json"), "w") as f:
            prompts_json = [{"index": i, "prompt": prompt} for i, prompt in enumerate(prompts)]
            json.dump({"data": prompts_json}, f, indent=4)

        return True

    def _samples_generation(self, num_samples: int, lm_names: List[str], vision_model: bool, device: str, time_performance: bool) -> bool:
        """
        Generate samples for the given prompts using the models and save them to a json file.
        The number of sample generated for each prompt is given by num_samples.

        :param num_samples: The number of samples to generate for each prompt.
        :type num_samples: int
        :param lm_names: The names of the models to be used for sampling.
        :type lm_names: List[str]
        :param vision_model: A flag indicating whether the models are vision models.
        :type vision_model: bool
        :param device: The Torch device to use for the operations.
        :type device: str
        :param time_performance: A flag indicating whether to measure the runtime of the operation.
        :type time_performance: bool
        :return: False if the models are not available or the prompts are missing, True otherwise.
        :rtype: bool
        """

        if self.lm is None:
            return False

        # Initialize logging
        logging.basicConfig(
            filename=os.path.join(self.workdir, "sample_log.log"),
            filemode="w",
            format="%(name)s - %(levelname)s - %(message)s",
            level=self.logging_level,
        )

        # Getting prompts from the json file
        with open(os.path.join(self.workdir, "prompts.json"), "r") as f:
            prompts = json.load(f)["data"]
        prompts = [p["prompt"] for p in prompts]

        if time_performance:
            with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                f.write("\n\nSample generation\n")

        # Sampling
        performance_times = []
        desc = "Language Models" if not vision_model else "Vision Models"
        for index, lm_name in (pbar := tqdm(enumerate(lm_names), desc=desc, leave=True, total=len(lm_names))):
            pbar.set_postfix_str(f"{lm_name}")
            logging.info(f"Running {lm_name}")

            self.lm[index].load_model(device=device)

            start = timer() if time_performance else None
            logging.info("Generating samples...")
            responses = []
            try:
                for p in tqdm(prompts, desc="Prompts", leave=False):
                    if vision_model:
                        local_response = self.lm[index].generate_image([p]*num_samples)
                    else:
                        local_response = self.lm[index].get_response_texts(
                                        self.lm[index].query(p, num_samples)
                                    )

                        logging.debug("Responses for the prompt are: ")
                        for r in local_response:
                            logging.debug(r)

                    responses.append(local_response)
            except Exception as e:
                logging.error(f"Error while running {lm_name}: {e}")
                print(f"Error while running {lm_name}: {e}")
                continue

            responses = self.parser.answer_parser(responses)

            end = timer() if time_performance else None
            performance_times.append(end - start if time_performance else None)
            if time_performance:
                with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                    f.write(f"\t - LM {lm_names[index]}: {end - start} seconds\n")

            # Save results to json files
            logging.info("Saving results...")
            if vision_model:
                for i, samples in enumerate(responses):
                    for j, sample in enumerate(samples):
                        sample.save(os.path.join(self.workdir, "images", f"{lm_name}_image_{i}_{j}.png"))
            else:
                with open(os.path.join(self.workdir, f"{lm_name}_samples.json"), "w") as f:
                    responses_json = [{"prompt_index": i, "samples": samples} for i, samples in enumerate(responses)]
                    json.dump({"data": responses_json}, f, indent=4)
                
            logging.info(f"Finished {lm_name}.")
            self.budget -= self.lm[index].cost

            logging.info(f"Remaining budget: {self.budget}")
            logging.info(f"used for lm: {self.lm[index].cost}")

            self.lm[index].unload_model()

            if self.budget < 0:
                break  
        
        logger = logging.getLogger("root")
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

        if time_performance:
            with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                f.write(f"\n\tTotal time: {sum(performance_times)} seconds\n")
                f.write(f"\tNumber of prompts per LM: {len(prompts)}\n")
                f.write(f"\tNumber of samples per prompt: {num_samples}\n")
        
        return True

    def _embeddings_generation(
            self,
            lm_names: List[str],
            embedding_lm_names: List[str],
            ground_truth: bool,
            vision_model: bool,
            num_samples: int,
            device: str,
            time_performance: bool
        ) -> bool:
        """
        Generate embeddings for the given samples using the embedding models and save them to a json file.

        :param lm_names: The names of the models used for sampling.
        :type lm_names: List[str]
        :param embedding_lm_names: The names of the embedding models used for the embedding.
        :type embedding_lm_names: List[str]
        :param ground_truth: A flag indicating whether to generate embeddings for the ground truth.
        :type ground_truth: bool
        :param vision_model: A flag indicating whether the models are vision models.
        :type vision_model: bool
        :param num_samples: The number of samples.
        :type num_samples: int
        :param device: The Torch device to use for the operations.
        :type device: str
        :param time_performance: A flag indicating whether to measure the runtime of the operation.
        :type time_performance: bool
        :return: False if the embedder or the embedding models are not available, True otherwise.
        :rtype: bool
        """

        embeddings_dir = os.path.join(self.workdir, "embeddings")

        if self.embedder is None or self.embedding_lm is None:
            return False
        
        if time_performance:
            with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                f.write("\n\nEmbedding generation\n")
        
        performance_times = []
        # Getting samples from the json file
        for index2, embedding_lm_name in (pbar := tqdm(enumerate(embedding_lm_names), desc="Embedding Language Models", leave=False, total=len(embedding_lm_names))):
            pbar.set_postfix_str(f"{embedding_lm_name}")

            try:
                self.embedding_lm[index2].load_model(device=device)
            except Exception as e:
                logging.error(f"Error while loading {embedding_lm_name}: {e}")
                print(f"Error while loading {embedding_lm_name}: {e}")
                continue
            # Initialize logging
            logging.basicConfig(
                filename=os.path.join(embeddings_dir, "embed_log.log"),
                filemode="w",
                format="%(name)s - %(levelname)s - %(message)s",
                level=self.logging_level,
            )
            
            if time_performance:
                with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                    f.write(f"\t - Embedding model: {embedding_lm_name}\n")

            embedding_times = []
            logging.info("Generating embeddings...")
            desc = "Language Models" if not vision_model else "Vision Models"
            for index, lm_name in (pbar2 := tqdm(enumerate(lm_names), desc=desc, leave=True, total=len(lm_names))):
                pbar2.set_postfix_str(f"{lm_names}")
                logging.info(f"Running {lm_names}...")
                start = timer() if time_performance else None

                if vision_model:
                    if not os.path.exists(os.path.join(self.workdir, "images", f"{lm_name}_image_0_0.png")):
                        return False
                    samples = []
                    try:
                        num_files = len(os.listdir(os.path.join(self.workdir, "images")))
                        for i in range(int(num_files / num_samples)):
                            sample = []
                            for j in range(num_samples):
                                sample.append(Image.open(os.path.join(self.workdir, "images", f"{lm_name}_image_{i}_{j}.png")))
                            samples.append(sample)
                    except Exception as e:
                        logging.error(f"Error while loading samples for {lm_name}: {e}")
                        print(f"Error while loading samples for {lm_name}: {e}")
                        continue
                else:
                    if not os.path.exists(os.path.join(self.workdir, f"{lm_name}_samples.json")):
                        return False

                    with open(os.path.join(self.workdir, f"{lm_name}_samples.json"), "r") as f:
                        samples = json.load(f)["data"]
                    samples = [s["samples"] for s in samples]

                embeddings = []
                for sample in tqdm(samples, desc="Prompts", leave=False):
                    embeddings.append(self.embedder.embed(self.embedding_lm[index2], sample))

                # Save results to json files
                logging.info("Saving results...")
                with open(os.path.join(embeddings_dir, f"{lm_name}_{embedding_lm_name}_embeddings.json"), "w") as f:
                    embeddings_json = [{"prompt_index": i, "embeddings": embedding} for i, embedding in enumerate(embeddings)]
                    json.dump({"data": embeddings_json}, f, indent=4)

                logging.info(f"Finished with {embedding_lm_name}-{lm_name}.")
                self.budget -= self.embedding_lm[index2].cost

                logging.info(f"Remaining budget: {self.budget}")
                logging.info(f"used for lm: {self.embedding_lm[index2].cost}")
                self.embedding_lm[index2].cost = 0

                end = timer() if time_performance else None
                embedding_times.append(end - start if time_performance else None)
                if time_performance:
                    with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                        f.write(f"\t\t - LM {lm_names[index]}: {embedding_times[-1]} seconds\n")

                if self.budget < 0:
                    break
            
            performance_times.append(sum(embedding_times) if time_performance else None)
            if time_performance:
                with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                    f.write(f"\t\t - Total time: {performance_times[-1]} seconds\n")

            self.embedding_lm[index2].unload_model()
        
        logger = logging.getLogger("root")
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

        if time_performance:
            with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                f.write(f"\n\tTotal time: {sum(performance_times)} seconds\n")
                f.write(f"\tNumber of language models per embedding: {len(lm_names)}\n")
                f.write(f"\tNumber of prompts per LM: {len(samples)}\n")
                f.write(f"\tNumber of samples per prompt: {len(samples[0])}\n")

        if ground_truth:
            # Getting the ground truth from the json file
            print("\n\nGenerating embeddings for the ground truth...")
            with open(os.path.join(self.workdir, "ground_truth.json"), "r") as f:
                ground_truth_data = json.load(f)["ground_truth"]

            # Initialize logging
            logging.basicConfig(
                filename=os.path.join(embeddings_dir, "ground_truth_log.log"),
                filemode="w",
                format="%(name)s - %(levelname)s - %(message)s",
                level=self.logging_level,
            )

            if time_performance and ground_truth:
                with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                    f.write("\n\nEmbedding generation for the ground truth\n")

            performance_times = []
            logging.info("Generating embeddings for ground of truth...")
            for index, embedding_lm_name in (pbar:= tqdm(enumerate(embedding_lm_names), desc="Embedding Language Models", leave=True, total=len(embedding_lm_names))):
                pbar.set_postfix_str(f"{embedding_lm_name}")
                logging.info(f"Running {embedding_lm_name}...")

                self.embedding_lm[index].load_model(device=device)
                start = timer() if time_performance else None

                embeddings = self.embedder.embed(self.embedding_lm[index], ground_truth_data)

                # Save results to json files
                logging.info("Saving results...")
                with open(os.path.join(embeddings_dir, f"ground_truth_{embedding_lm_name}_embeddings.json"), "w") as f:
                    embeddings_json = [{"ground_truth_index": i, "embeddings": embedding} for i, embedding in enumerate(embeddings)]
                    json.dump({"data": embeddings_json}, f, indent=4)

                logging.info(f"Finished with {embedding_lm_name}.")
                self.budget -= self.embedding_lm[index].cost

                logging.info(f"Remaining budget: {self.budget}")
                logging.info(f"used for lm: {self.embedding_lm[index].cost}")

                end = timer() if time_performance else None
                performance_times.append(end - start if time_performance else None)
                if time_performance and ground_truth:
                    with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                        f.write(f"\t - Embedding model {embedding_lm_names[index]}: {end - start} seconds\n")

                self.embedding_lm[index].unload_model()

                if self.budget < 0:
                    break
        
        logger = logging.getLogger("root")
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

        if time_performance and ground_truth:
            with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                f.write(f"\n\tTotal time: {sum(performance_times)} seconds\n")
                f.write(f"\tNumber of ground truth samples per embedding: {len(ground_truth_data)}\n")

        return True

    def _operations(self, ground_truth: bool, time_performance: bool) -> None:
        """
        Execute the operations in the given order.
        If an operation fails, it will be logged and the scheduler will continue with the next operation.
        logging_level and ground_truth are passed as custom inputs to the operations.

        :param ground_truth: A flag indicating whether the ground truth is available.
        :type ground_truth: bool
        :param time_performance: A flag indicating whether to measure the runtime of the operations.
        :type time_performance: bool
        """
        if time_performance:
            with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                f.write("\n\nOperations\n")
        
        performance_times = []
        for operation in self.operations:
            try:
                start = timer() if time_performance else None
                operation.execute(custom_inputs={"logging_level": self.logging_level, "ground_truth": ground_truth})
                end = timer() if time_performance else None
                performance_times.append(end - start if time_performance else None)
                if time_performance:
                    with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                        f.write(f"\t - Operation {operation}: {end - start} seconds\n")

                print(f"Done! Remaining {len(self.operations) - self.operations.index(operation) - 1} operations to run\n")

                logger = logging.getLogger("root")
                for handler in logger.handlers:
                    handler.close()
                    logger.removeHandler(handler)
            except Exception as e:
                end = timer() if time_performance else None
                logging.error(f"Error while running {operation}: {e}")
                continue

        if time_performance:
            with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                f.write(f"\n\tTotal time: {sum(performance_times)} seconds\n")

    def run(
            self,
            startingPoint: StartingPoint = StartingPoint.PROMPT,
            defaultDirectories: bool = True,
            bertScore: bool = False,
            selfCheckGPT: bool = False,
            llm_as_a_judge: bool = False,
            vision: bool = False,
            checkEmbed: bool = True,
            ground_truth: bool = False,
            spacy_separator: bool = True,
            time_performance: bool = False,
            rebase_results: bool = False,
            reference_text: bool = False,
            num_samples: int = 10, 
            lm_names: List[str] = [],
            embedding_lm_names: List[str] = [],
            bertScore_model: str = None,
            batch_size: int = 64,
            device: str = None
        ) -> None:
        """
        Run the scheduler starting from the given starting point and execute the operations in the given order.

        :param startingPoint: The starting point for the scheduler. Defaults to StartingPoint.PROMPT.
        :type startingPoint: StartingPoint
        :param defaultDirectories: A flag indicating whether to create the default directories for the generated data. If False, the directories must be created manually. Defaults to True.
        :type defaultDirectories: bool
        :param bertScore: A flag indicating whether to execute the BERTScore operation. Defaults to False.
        :type bertScore: bool
        :param selfCheckGPT: A flag indicating whether to execute the selfCheckGPT operation. Defaults to False.
        :type selfCheckGPT: bool
        :param llm_as_a_judge: A flag indicating whether to execute the LLM as a judge operation. Defaults to False.
        :type llm_as_a_judge: bool
        :param vision: A flag indicating whether to execute the vision operation. Defaults to False.
        :type vision: bool
        :param checkEmbed: A flag indicating whether to execute the CheckEmbed operation. Defaults to True.
        :type checkEmbed: bool
        :param ground_truth: A flag indicating whether ground truth is available. Defaults to False.
        :type ground_truth: bool
        :param spacy_separator: A flag indicating whether to use the spacy separator for the SelfCheckGPT operation. If False, sentences are separated at the newline character. Defaults to True.
        :type spacy_separator: bool
        :param time_performance: A flag indicating whether to measure the runtime of the operations. Defaults to False.
        :type time_performance: bool
        :param rebase_results: A flag indicating whether to rebase the results of the CheckEmbed operation. Defaults to False.
        :type rebase_results: bool
        :param reference_text: A flag indicating whether to use the reference text for the SelfCheckGPT operation. Defaults to False.
        :type reference_text: bool
        :param num_samples: The number of samples to generate for each prompt. Defaults to 10.
        :type num_samples: int
        :param lm_names: Overwrite default names of used LLMs. Defaults to an empty list.
        :type lm_names: List[str]
        :param embedding_lm_names: Overwrite default names of used embedding models. Defaults to an empty list.
        :type embedding_lm_names: List[str]
        :param bertScore_model: The BERTScore model to be used for the operation. Defaults to None.
        :type bertScore_model: str
        :param batch_size: The batch size for the operations (needed for BertScore and SelfCheckGPT). Defaults to 64.
        :type batch_size: int
        :param device: The Torch device to use for the operations. Defaults to None.
        :type device: str
        """

        # Create the directory structure if necessary
        if defaultDirectories:
            os.makedirs(os.path.join(self.workdir, "embeddings"), exist_ok=True)
            if checkEmbed:
                os.makedirs(os.path.join(self.workdir, "CheckEmbed"), exist_ok=True)
            if bertScore:
                os.makedirs(os.path.join(self.workdir, "BertScore"), exist_ok=True)
            if selfCheckGPT:
                os.makedirs(os.path.join(self.workdir, "SelfCheckGPT"), exist_ok=True)
            if llm_as_a_judge:
                os.makedirs(os.path.join(self.workdir, "Judge"), exist_ok=True)
            if vision:
                os.makedirs(os.path.join(self.workdir, "images"), exist_ok=True)
                os.makedirs(os.path.join(self.workdir, "vision"), exist_ok=True)

        if time_performance:
            if not os.path.exists(os.path.join(self.workdir, "runtimes")):
                os.mkdir(os.path.join(self.workdir, "runtimes"))
            with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                f.write("Starting performance measurement\n")

        # Set the default names if not provided
        lm_names = lm_names if len(lm_names) > 0 else [lm.name for lm in self.lm]
        embedding_lm_names = embedding_lm_names if len(embedding_lm_names) > 0 else [embedding_lm.name for embedding_lm in self.embedding_lm]

        print("Starting point: ", startingPoint)

        # PROMPT GENERATION
        if startingPoint.value <= StartingPoint.PROMPT.value:
            start = timer() if time_performance else None

            executed = self._prompt_generation()
            if not executed:
                print("For prompt generation a parser is required")
                return

            print("\n\nPrompt generation completed")

            if ground_truth:
                if self.parser is None:
                    if not os.path.exists(os.path.join(self.workdir, "ground_truth.json")):
                        print("For the ground truth generation a parser is required")
                else:
                    with open(os.path.join(self.workdir, "ground_truth.json"), "w") as f:
                        json.dump({"ground_truth": self.parser.ground_truth_extraction()}, f, indent=4)
                    print("Ground truth generation completed\n")

            end = timer() if time_performance else None
            if time_performance:
                with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                    f.write(f"\n\nPrompt generation took {end - start} seconds\n")

        # SAMPLES GENERATION
        if startingPoint.value <= StartingPoint.SAMPLES.value:
            print("\n\nStarting samples generation...")
            executed = self._samples_generation(num_samples, lm_names, vision, device, time_performance)
            if not executed:
                print("For samples generation a parser or a language model with prompts.json file is required")
                return
            
            print("Remaining budget: ", self.budget)

        # EMBEDDINGS GENERATION
        if startingPoint.value <= StartingPoint.EMBEDDINGS.value:
            print("\n\nEmbeddings generation started...")
            executed = self._embeddings_generation(lm_names, embedding_lm_names, ground_truth, vision, num_samples, device, time_performance)
            if not executed:
                print("For the embeddings generation requires an embedder, an embedding model, a lists of LLMs and embedding model names.")
                return
            
            print("Remaining budget: ", self.budget)

        print("\n\nStarting operations...")
        if selfCheckGPT:
            for selfcheckgpt in self.selfCheckGPTOperation:
                selfcheckgpt.execute(custom_inputs={"logging_level": self.logging_level, "lm_names": lm_names, "device": device, "batch_size": batch_size, "spacy": spacy_separator, "time_performance": time_performance, "reference_text": reference_text})
            print("Done!\n")

        if bertScore:
            self.bertScoreOperation.execute(custom_inputs={"logging_level": self.logging_level, "ground_truth": ground_truth, "model_type": bertScore_model, "lm_names": lm_names, "batch_size": batch_size, "device": device, "time_performance": time_performance})
            print("Done!\n")

        if checkEmbed:
            print("\n\nStarting CheckEmbed operation...")
            self.checkEmbedOperation.execute(custom_inputs={"ground_truth": ground_truth, "time_performance": time_performance, "rebase_results": rebase_results})  
            print("Done!\n")

        if llm_as_a_judge:
            if self.llm_as_a_judge_Operation is None:
                print("For the LLM as a judge operation a LLMAsAJudgeOperation instance is required")
                return
            if len(self.llm_as_a_judge_models) == 0:
                print("For the LLM as a judge operation a list of LLMs is required")
                return
            print("\n\nStarting LLM as a judge operation...")
            for llm_as_a_judge_model in self.llm_as_a_judge_models:
                self.llm_as_a_judge_Operation.execute(custom_inputs={"model": llm_as_a_judge_model})
            print("Done!\n")        

        print("\n\nStarting other operations...")
        self._operations(ground_truth=ground_truth, time_performance=time_performance)
        print("\nOperations completed!")
