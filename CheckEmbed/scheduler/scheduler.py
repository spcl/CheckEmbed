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
from typing import List
from timeit import default_timer as timer
from tqdm import tqdm

from CheckEmbed.language_models import AbstractLanguageModel
from CheckEmbed.embedding_models import AbstractEmbeddingModel
from CheckEmbed.operations import BertScoreOperation
from CheckEmbed.operations import CheckEmbedOperation
from CheckEmbed.operations import SelfCheckGPTOperation
from CheckEmbed.operations import Operation
from CheckEmbed.parser import Parser
from CheckEmbed.embedder import Embedder

from enum import Enum

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
            lm: List[AbstractLanguageModel] = None, 
            embedding_lm: List[AbstractEmbeddingModel] = None,
            operations: List[Operation] = [],
            bertScoreOperation: BertScoreOperation = None,
            selfCheckGPTOperation: SelfCheckGPTOperation = None,
            checkEmbedOperation: CheckEmbedOperation = None,
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
        :param lm: A list of AbstractLanguageModel instances representing the language models used for sampling. Defaults to None.
        :type lm: List[AbstractLanguageModel]
        :param embedding_lm: A list of AbstractEmbeddingModel instances representing the embedding models used for the embedding generation. Defaults to None.
        :type embedding_lm: List[AbstractEmbeddingModel]
        :param operations: A list of Operation instances representing additional operations to be executed. Defaults to an empty list.
        :type operations: List[Operation]
        :param bertScoreOperation: An instance of a custom BertScoreOperation class for the BERTScore computation. Defaults to None. If None, the default BertScoreOperation will be used.
        :type bertScoreOperation: BertScoreOperation
        :param selfCheckGPTOperation: An instance of a custom selfCheckGPTOperation class for the SelfCheckGPT computation. Defaults to None. If None, the default SelfCheckGPTOperation will be used.
        :type selfCheckGPTOperation: selfCheckGPTOperation
        :param checkEmbedOperation: An instance of a custom CheckEmbedOperation class for CheckEmbed computation. Defaults to None. If None, the default CheckEmbedOperation will be used.
        :type checkEmbedOperation: CheckEmbedOperation
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
        self.selfCheckGPTOperation = SelfCheckGPTOperation(os.path.join(workdir, "SelfCheckGPT"), workdir) if selfCheckGPTOperation is None else selfCheckGPTOperation
        self.checkEmbedOperation = CheckEmbedOperation(os.path.join(workdir, "CheckEmbed"), os.path.join(workdir, "embeddings")) if checkEmbedOperation is None else checkEmbedOperation

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

    def _samples_generation(self, num_sample: int, lm_names: List[str], device: str, time_performance: bool) -> bool:
        """
        Generate samples for the given prompts using the language models and save them to a json file.
        The number of sample generated for each prompt is given by num_sample.

        :param num_sample: The number of samples to generate for each prompt.
        :type num_sample: int
        :param lm_names: The names of the language models to be used for sampling.
        :type lm_names: List[str]
        :param device: The Torch device to use for the operations.
        :type device: str
        :param time_performance: A flag indicating whether to measure the time performance of the operation.
        :type time_performance: bool
        :return: False if the language models are not available or the prompts are missing, True otherwise.
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
                f.write(f"\n\nSample generation\n")

        # Sampling
        performance_times = []
        for index, lm_name in (pbar := tqdm(enumerate(lm_names), desc="Language Models", leave=True, total=len(lm_names))):
            pbar.set_postfix_str(f"{lm_name}")
            logging.info(f"Running {lm_name}")

            self.lm[index].load_model(device=device)

            start = timer() if time_performance else None
            logging.info("Generating samples...")
            responses = []
            try:
                for p in tqdm(prompts, desc="Prompts", leave=False):
                    local_response = self.lm[index].get_response_texts(
                                    self.lm[index].query(p, num_sample)
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
                f.write(f"\tNumber of samples per prompt: {num_sample}\n")
        
        return True

    def _embeddings_generation(
            self,
            lm_names: List[str],
            embedding_lm_names: List[str],
            ground_truth: bool,
            device: str,
            time_performance: bool
        ) -> bool:
        """
        Generate embeddings for the given samples using the embedding models and save them to a json file.

        :param lm_names: The names of the language models used for sampling.
        :type lm_names: List[str]
        :param embedding_lm_names: The names of the embedding models used for the embedding.
        :type embedding_lm_names: List[str]
        :param ground_truth: A flag indicating whether to generate embeddings for the ground truth.
        :type ground_truth: bool
        :param device: The Torch device to use for the operations.
        :type device: str
        :param time_performance: A flag indicating whether to measure the time performance of the operation.
        :type time_performance: bool
        :return: False if the embedder or the embedding models are not available, True otherwise.
        :rtype: bool
        """

        embeddings_dir = os.path.join(self.workdir, "embeddings")

        if self.embedder is None or self.embedding_lm is None:
            return False
        
        if time_performance:
            with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                f.write(f"\n\nEmbedding generation\n")
        
        performance_times = []
        # Getting samples from the json file
        for index2, embedding_lm_name in (pbar := tqdm(enumerate(embedding_lm_names), desc="Embedding Language Models", leave=False, total=len(embedding_lm_names))):
            pbar.set_postfix_str(f"{embedding_lm_name}")

            self.embedding_lm[index2].load_model(device=device)
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
            for index, lm_name in (pbar2 := tqdm(enumerate(lm_names), desc="Language Models", leave=True, total=len(lm_names))):
                pbar2.set_postfix_str(f"{lm_names}")
                logging.info(f"Running {lm_names}...")
                start = timer() if time_performance else None

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

                logging.info(f"Finished with {embedding_lm_name}.")
                self.budget -= self.embedding_lm[index2].cost

                logging.info(f"Remaining budget: {self.budget}")
                logging.info(f"used for lm: {self.embedding_lm[index2].cost}")

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
                    f.write(f"\n\nEmbedding generation for the ground truth\n")

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
        :param time_performance: A flag indicating whether to measure the time performance of the operations.
        :type time_performance: bool
        """
        if time_performance:
            with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                f.write(f"\n\nOperations\n")
        
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
            checkEmbed: bool = True,
            ground_truth: bool = False,
            spacy_separator: bool = True,
            time_performance: bool = False,
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
        :param checkEmbed: A flag indicating whether to execute the CheckEmbed operation. Defaults to True.
        :type checkEmbed: bool
        :param ground_truth: A flag indicating whether ground truth is available. Defaults to False.
        :type ground_truth: bool
        :param spacy_separator: A flag indicating whether to use the spacy separator for the SelfCheckGPT operation. If False, sentences are separated at the newline character. Defaults to True.
        :type spacy_separator: bool
        :param time_performance: A flag indicating whether to measure the time performance of the operations. Defaults to False.
        :type time_performance: bool
        :param num_samples: The number of samples to generate for each prompt. Defaults to 10.
        :type num_samples: int
        :param lm_names: The names of the language models used for sampling. Defaults to an empty list.
        :type lm_names: List[str]
        :param embedding_lm_names: The names of the embedding models used for the embedding. Defaults to an empty list.
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
            if not os.path.exists(os.path.join(self.workdir, "embeddings")):
                os.mkdir(os.path.join(self.workdir, "embeddings"))

            if not os.path.exists(os.path.join(self.workdir, "plots")):
                os.mkdir(os.path.join(self.workdir, "plots"))

            if not os.path.exists(os.path.join(self.workdir, "CheckEmbed")):
                os.mkdir(os.path.join(self.workdir, "CheckEmbed"))

            if not os.path.exists(os.path.join(self.workdir, "plots", "CheckEmbed")):
                os.mkdir(os.path.join(self.workdir, "plots", "CheckEmbed"))

            if bertScore and not os.path.exists(os.path.join(self.workdir, "BertScore")):
                os.mkdir(os.path.join(self.workdir, "BertScore"))
            
            if bertScore and not os.path.exists(os.path.join(self.workdir, "plots", "BertScore")):
                os.mkdir(os.path.join(self.workdir, "plots", "BertScore"))

            if selfCheckGPT and not os.path.exists(os.path.join(self.workdir, "SelfCheckGPT")):
                os.mkdir(os.path.join(self.workdir, "SelfCheckGPT"))

            if selfCheckGPT and not os.path.exists(os.path.join(self.workdir, "plots", "SelfCheckGPT")):
                os.mkdir(os.path.join(self.workdir, "plots", "SelfCheckGPT"))

        if time_performance:
            if not os.path.exists(os.path.join(self.workdir, "runtimes")):
                os.mkdir(os.path.join(self.workdir, "runtimes"))
            with open(os.path.join(self.workdir, "runtimes", "performance_log.log"), "a") as f:
                f.write("Starting performance measurement\n")

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
            executed = self._samples_generation(num_samples, lm_names, device, time_performance)
            if not executed:
                print("For samples generation a parser or a language model with prompts.json file is required")
                return
            
            print("Remaining budget: ", self.budget)

        # EMBEDDINGS GENERATION
        if startingPoint.value <= StartingPoint.EMBEDDINGS.value:
            print("\n\nEmbeddings generation started...")
            executed = self._embeddings_generation(lm_names, embedding_lm_names, ground_truth, device, time_performance)
            if not executed:
                print("For the embeddings generation requires an embedder, an embedding model, a lists of LLMs and embedding model names.")
                return
            
            print("Remaining budget: ", self.budget)

        print("\n\nStarting operations...")
        if bertScore:
            self.bertScoreOperation.execute(custom_inputs={"logging_level": self.logging_level, "ground_truth": ground_truth, "model_type": bertScore_model, "lm_names": lm_names, "batch_size": batch_size, "device": device, "time_performance": time_performance})
            print(f"Done!\n")

        if selfCheckGPT:
            self.selfCheckGPTOperation.execute(custom_inputs={"logging_level": self.logging_level, "lm_names": lm_names, "device": device, "batch_size": batch_size, "spacy": spacy_separator, "time_performance": time_performance})
            print(f"Done!\n")

        if checkEmbed:
            print("\n\nStarting CheckEmbed operation...")
            self.checkEmbedOperation.execute(custom_inputs={"ground_truth": ground_truth, "time_performance": time_performance})  
            print(f"Done!\n")

        print("\n\nStarting other operations...")
        self._operations(ground_truth=ground_truth, time_performance=time_performance)
        print("\nOperations completed!")
