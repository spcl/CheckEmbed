# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import os
import gc
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
from huggingface_hub import snapshot_download

from typing import List, Literal, Union

from CheckEmbed.embedding_models import AbstractEmbeddingModel


class Stella(AbstractEmbeddingModel):
    """
    The Stella class handles interactions with the Stella embedding model family using the provided configuration.
    
    Inherits from the AbstractEmbeddingModel class and implements its abstract methods.
    """

    def __init__(
        self, model_name: str = "", variant: Literal["400M-v5", "1.5B-v5", ""] = "400M-v5", name: str = "stella-en-", cache: bool = False, max_length: int = 4096, batch_size: int = 64
    ) -> None:
        """
        Initialize the Stella instance with configuration, model details, and caching options.
        
        :param model_name: Name of the model, default is "". Used to select the correct configuration.
        :type model_name: str
        :param variant: The variant of the Stella model to use. Defaults to "400M_v5".
        :type variant: Literal["400M-v5", "1.5B-v5", ""]
        :param name: Name used for output files. Defaults to "stella-en-".
        :type name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        :param max_length: The maximum length of the input text. Defaults to 4096.
        :type max_length: int
        :param batch_size: The batch size to be used for the model. Defaults to 64.
        :type batch_size: int
        """
        super().__init__(model_name=model_name, name=name + variant, cache=cache)
        self.max_length = max_length
        self.batch_size = batch_size

    def load_model(self, device: str = None) -> None:
        """
        Load the model and tokenizer based on the given model name.
        
        :param device: The device to load the model on.
        :type device: str
        """
        try:
            model_dir = snapshot_download(repo_id=self.model_name)
        except Exception as e:
            raise ValueError(f"Model {self.model_name} not found in the Hugging Face Hub") from e
        vector_linear_directory = f"2_Dense_{self.max_length}"
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.vector_linear = torch.nn.Linear(in_features=self.model.config.hidden_size, out_features=self.max_length)
        vector_linear_dict = {
            k.replace("linear.", ""): v for k, v in
            torch.load(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin")).items()
        }
        self.vector_linear.load_state_dict(vector_linear_dict)
        self.vector_linear.to(device)
        
    def unload_model(self) -> None:
        """
        Unload the model and tokenizer from memory.
        """
        del self.model
        
        gc.collect()
        torch.cuda.empty_cache()

        self.model = None

    def generate_embedding(self, input: Union[List[str], str]) -> List[List[float]]:
        """
        Generate the embeddings for the given input text.
        
        :param input: The input text to embed.
        :type input: Union[List[str], str]
        :return: The embeddings of the text.
        :rtype: List[List[float]]
        """
        if isinstance(input, str):
            input = [input]
        
        total_embeddings = []
        flag = True

        while flag:
            try:
                batched_responses = [input[i:i+self.batch_size] for i in range(0, len(input), self.batch_size)]

                batch_dict = None
                attention_mask = None
                last_hidden_state = None
                last_hidden = None
                docs_vectors = None

                for batch in tqdm(batched_responses, desc="Batches to Embed", leave=False, total=len(batched_responses)):
                    with torch.no_grad():
                        batch_dict = self.tokenizer(batch, padding="longest", truncation=True, max_length=512, return_tensors="pt")
                        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
                        attention_mask = batch_dict["attention_mask"]
                        last_hidden_state = self.model(**batch_dict)[0]
                        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
                        docs_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                        docs_vectors = normalize(self.vector_linear(docs_vectors).cpu().detach().numpy())

                    total_embeddings.extend(docs_vectors.tolist())

                    del batch_dict, attention_mask, last_hidden_state, last_hidden, docs_vectors
                    gc.collect()
                    torch.cuda.empty_cache()

                flag = False
            
            except Exception as e:
                batch_dict = None
                attention_mask = None
                last_hidden_state = None
                last_hidden = None
                docs_vectors = None
                total_embeddings = []
                gc.collect()
                torch.cuda.empty_cache()

                print("Error occurred, reducing batch size and retrying")
                if self.batch_size == 1:
                    raise e
                self.batch_size = self.batch_size // 2

        return total_embeddings
