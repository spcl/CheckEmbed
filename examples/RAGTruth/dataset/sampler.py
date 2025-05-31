# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from openai import OpenAI
from typing import List, Tuple
from vllm import LLM, SamplingParams

client = OpenAI(
  api_key='sk-',
  organization='org-'
)

def get_gpt4_answer(prompt: str, temperature: float, samples: int, llm: TODO) -> Tuple[List[str], float]:
    """
    Get responses from GPT4 concurrently for the given prompt.

    :param prompt: Prompt used to query the language model.
    :type prompt: str
    :param temperature: Temperature for the language model.
    :type temperature: float
    :param samples: Number of responses to generate.
    :type samples: int
    :param llm: Language model to use. Parameter is not used for this function.
    :type llm: TODO
    :return: Tuple of the list of responses and the total cost.
    :rtype: Tuple[List[str], float]
    """

    def get_res(message: str) -> Tuple[str, float]:
        """
        Get a single response from GPT4 for the given prompt.

        :param message: Prompt to submit.
        :type message: str
        :return: Tuple of the returned response and the cost its generation.
        :rtype: Tuple[str, float]
        """
        while True:
            try:
                res = client.chat.completions.create(
                    model="gpt-4-0613",
                    messages=message,
                    temperature=temperature,
                )
                break
            except openai.error.RateLimitError:
                print('openai.error.RateLimitError\nRetrying...')
                time.sleep(60)
            except openai.error.ServiceUnavailableError:
                print('openai.error.ServiceUnavailableError\nRetrying...')
                time.sleep(20)
            except openai.error.Timeout:
                print('openai.error.Timeout\nRetrying...')
                time.sleep(20)
            except openai.error.APIError:
                print('openai.error.APIError\nRetrying...')
                time.sleep(20)
            except openai.error.APIConnectionError:
                print('openai.error.APIConnectionError\nRetrying...')
                time.sleep(20)
            
        prompt_tokens = res.usage.prompt_tokens
        completion_tokens = res.usage.completion_tokens
        cost = (0.03 * prompt_tokens / 1000.0 + 0.06 * completion_tokens / 1000.0)
        return res.choices[0].message.content, cost
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(get_res, [{"role": "user", "content": prompt}]) for _ in range(samples)]
        results = []
        total_cost = 0
        for future in as_completed(futures):
            response, cost = future.result()
            total_cost += cost
            results.append(response)    
    
    return results, total_cost

def get_gpt3_5_turbo_answer(prompt: str, temperature: float, samples: int, llm: TODO) -> Tuple[List[str], float]:
    """
    Get responses from GPT3.5-Turbo concurrently for the given prompt.

    :param prompt: Prompt used to query the language model.
    :type prompt: str
    :param temperature: Temperature for the language model.
    :type temperature: float
    :param samples: Number of responses to generate.
    :type samples: int
    :param llm: Language model to use. Parameter is not used for this function.
    :type llm: TODO
    :return: Tuple of the list of responses and the total cost.
    :rtype: Tuple[List[str], float]
    """

    def get_res(message: str) -> Tuple[str, float]:
        """
        Get a single response from GPT3.5-Turbo for the given prompt.

        :param message: Prompt to submit.
        :type message: str
        :return: Tuple of the returned response and the cost its generation.
        :rtype: Tuple[str, float]
        """
        while True:
            try:
                res = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=message,
                    temperature=temperature,
                )
                break
            except openai.error.RateLimitError:
                print('openai.error.RateLimitError\nRetrying...')
                time.sleep(60)
            except openai.error.ServiceUnavailableError:
                print('openai.error.ServiceUnavailableError\nRetrying...')
                time.sleep(20)
            except openai.error.Timeout:
                print('openai.error.Timeout\nRetrying...')
                time.sleep(20)
            except openai.error.APIError:
                print('openai.error.APIError\nRetrying...')
                time.sleep(20)
            except openai.error.APIConnectionError:
                print('openai.error.APIConnectionError\nRetrying...')
                time.sleep(20)
            
        prompt_tokens = res.usage.prompt_tokens
        completion_tokens = res.usage.completion_tokens
        cost = (0.0005 * prompt_tokens / 1000.0 + 0.0015 * completion_tokens / 1000.0)
        return res.choices[0].message.content, cost
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(get_res, [{"role": "user", "content": prompt}]) for _ in range(samples)]
        results = []
        total_cost = 0
        for future in as_completed(futures):
            response, cost = future.result()
            total_cost += cost
            results.append(response)    
    
    return results, total_cost

def get_mistral_answer(prompt: str, temperature: float, samples: int, llm: TODO) -> Tuple[List[str], float]:
    """
    Get responses from a Mistral model for the given prompt.

    :param prompt: Prompt used to query the language model.
    :type prompt: str
    :param temperature: Temperature for the language model.
    :type temperature: float
    :param samples: Number of responses to generate.
    :type samples: int
    :param llm: Language model to use.
    :type llm: TODO
    :return: Tuple of the list of responses and the total cost.
    :rtype: Tuple[List[str], float]
    """
    prompts = [prompt] * samples

    sampling_params = SamplingParams(temperature=temperature, top_p=1, max_tokens=8192)

    outputs = llm.generate(prompts, sampling_params)

    texts = []
    for output in outputs:
        texts.append(output.outputs[0].text)
    return texts, 0.0

def get_llama_answer(prompt: str, temperature: float, samples: int, llm: TODO) -> Tuple[List[str], float]:
    """
    Get responses from a LLaMA model for the given prompt.

    :param prompt: Prompt used to query the language model.
    :type prompt: str
    :param temperature: Temperature for the language model.
    :type temperature: float
    :param samples: Number of responses to generate.
    :type samples: int
    :param llm: Language model to use.
    :type llm: TODO
    :return: Tuple of the list of responses and the total cost.
    :rtype: Tuple[List[str], float]
    """
    prompts = [prompt]*samples

    sampling_params = SamplingParams(temperature=temperature, top_p=1, max_tokens=8192)

    outputs = llm.generate(prompts, sampling_params)

    # Append the output
    texts = []
    for output in outputs:
        texts.append(output.outputs[0].text)

    return texts, 0.0


name_model_dict = {
    "gpt-4-0613": get_gpt4_answer,
    "gpt-3.5-turbo-0613": get_gpt3_5_turbo_answer,
    "mistral-7B-instruct": get_mistral_answer,
    "llama-2-7b-chat": get_llama_answer,
    "llama-2-13b-chat": get_llama_answer,
    "llama-2-70b-chat": get_llama_answer,
}
model_name_path = {
    "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama-2-70b-chat": "TheBloke/Llama-2-70B-Chat-AWQ",
    "mistral-7B-instruct": "mistralai/Mistral-7B-Instruct-v0.1"
}

def main():
    """
    Run the sample generation loop.
    """
    budget = 100

    model_temp = None
    with open("response.json", "r") as f:
        model_temp = json.load(f)

    model_temp = sorted(model_temp, key=lambda x: x["model"])
    
    prompt_soource = None
    with open("source_info.json", "r") as f:
        prompt_soource = json.load(f)

    name_model = ""
    llm = None
    for i, element in enumerate(model_temp):
        function_call = name_model_dict[element["model"]]
        temperature = element["temperature"]
        source_id = element["source_id"]
        id = element["id"]
        prompt = ""
        for source in prompt_soource:
            if source["source_id"] == source_id:
                prompt = source["prompt"]
                break
        
        if element["model"] != name_model:
            name_model = element["model"]
            if name_model in model_name_path:
                if "mistral" in name_model:
                    llm = LLM(
                        model=model_name_path[name_model],
                        tensor_parallel_size=2,
                        tokenizer_mode="mistral",
                        enforce_eager=True
                    )
                else:
                    llm = LLM(
                        model=model_name_path[name_model],
                        quantization="awq" if name_model == "llama-2-70b-chat" else None,
                        tensor_parallel_size=2 if name_model == "llama-2-7b-chat" else 4,
                        enforce_eager=True
                    )
            else:
                llm = None
        
        result, cost = function_call(prompt, temperature, 10, llm)

        with open("samples.json", "a+") as f:
            json_records = json.dumps({"id": id, "model": element["model"], "temperature": temperature, "source_id": source_id, "result": result})
            f.write(json_records + "\n")

        budget -= cost
        if budget < 0:
            break
        if i % 50 == 0:
            print(f"Budget left: {budget}")

if __name__ == '__main__':
    main()
    ### CURRENTLY NOT WORKING CORRECTLY ###
    ### WHEN USING VLLM WITH MULTI TENSOR PARALLELISM, MODEL WILL NOT BE ABLE TO SWITCH ###
    ### SO it can be used, but if you go from mistral to llama, you will have to restart the script ###
