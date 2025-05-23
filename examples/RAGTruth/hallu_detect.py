# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari
#
# Most of the code below is from the HalluDetect repository.
# https://github.com/Baylor-AI/HalluDetect
#
# Released under the MIT License.

import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    LEDForConditionalGeneration,
    LEDTokenizer,
)

#############################
## LLM Model Class Generic ##
#############################

class LLMModel:
    def __init__(self, device):
        self.device = device
        pass

    def getName(self) -> str:
        return self.model_name

    def getSanitizedName(self) -> str:
        return self.model_name.replace("/", "__")

    def generate(self, inpt):
        pass

    ##Move in future commits this method to an utils.py
    def truncate_string_by_len(self, s, truncate_len):
        words = s.split()
        truncated_words = words[:-truncate_len] if truncate_len > 0 else words
        return " ".join(truncated_words)

    # Method to get the vocabulary probabilities of the LLM for a given token on the generated text from LLM-Generator
    def getVocabProbsAtPos(self, pos, token_probs):
        sorted_probs, sorted_indices = torch.sort(token_probs[pos, :], descending=True)
        return sorted_probs

    def getMaxLength(self):
        return self.model.config.max_position_embeddings

    # By default knowledge is the empty string. If you want to add extra knowledge you can do it like in the cases of the qa_data.json and dialogue_data.json
    def extractFeatures(
        self,
        knowledge="",
        conditionted_text="",
        generated_text="",
        features_to_extract={},
    ):
        self.model.eval()

        # Also in the case of the LED model, there is no need to truncate the text in the context of this dataset.
        total_len = len(knowledge) + len(conditionted_text) + len(generated_text)
        truncate_len = min(total_len - self.tokenizer.model_max_length, 0)

        # Truncate knowledge in case is too large
        knowledge = self.truncate_string_by_len(knowledge, truncate_len // 2)
        # Truncate text_A in case is too large
        conditionted_text = self.truncate_string_by_len(
            conditionted_text, truncate_len - (truncate_len // 2)
        )

        if self.device == 'cuda':
            self.device = 'cuda:0'

        inputs = self.tokenizer(
            [knowledge + conditionted_text + generated_text],
            return_tensors="pt",
            max_length=self.getMaxLength(),
            truncation=True,
        )

        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = F.softmax(logits, dim=-1)
        probs = probs.to(self.device)
        
        tokens_generated_length = len(self.tokenizer.tokenize(generated_text))
        start_index = logits.shape[1] - tokens_generated_length
        conditional_probs = probs[0, start_index :]

        token_ids_generated = inputs["input_ids"][0, start_index :].tolist()
        token_probs_generated = [
            conditional_probs[i, tid].item()
            for i, tid in enumerate(token_ids_generated)
        ]

        minimum_token_prob = min(token_probs_generated)
        average_token_prob = sum(token_probs_generated) / len(token_probs_generated)

        maximum_diff_with_vocab = -1
        minimum_vocab_extreme_diff = 100000000000

        if features_to_extract["MDVTP"] is True or features_to_extract["MMDVP"] is True:
            size = len(token_probs_generated)
            for pos in range(size):
                vocabProbs = self.getVocabProbsAtPos(pos, conditional_probs)
                maximum_diff_with_vocab = max(
                    [
                        maximum_diff_with_vocab,
                        self.getDiffVocab(vocabProbs, token_probs_generated[pos]),
                    ]
                )
                minimum_vocab_extreme_diff = min(
                    [
                        minimum_vocab_extreme_diff,
                        self.getDiffMaximumWithMinimum(vocabProbs),
                    ]
                )

        # allFeatures = [minimum_token_prob, average_token_prob, maximum_diff_with_vocab, minimum_vocab_extreme_diff]

        allFeatures = {
            "mtp": minimum_token_prob,
            "avgtp": average_token_prob,
            "MDVTP": maximum_diff_with_vocab,
            "MMDVP": minimum_vocab_extreme_diff,
        }

        selectedFeatures = {}
        for key, feature in features_to_extract.items():
            if feature is True:
                selectedFeatures[key] = allFeatures[key]

        return selectedFeatures

    def getDiffVocab(self, vocabProbs, tprob):
        return (vocabProbs[0] - tprob).item()

    def getDiffMaximumWithMinimum(self, vocabProbs):
        return (vocabProbs[0] - vocabProbs[-1]).item()
    
class Gemma(LLMModel):
    def __init__(self, device):
        self.model_name = "google/gemma-7b-it"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        super().__init__(device)


    def generate(self, inpt):
        inputs = self.tokenizer(
            [inpt], max_length=self.getMaxLength(), return_tensors="pt", truncation=True
        )
        summary_ids = self.model.generate(inputs["input_ids"])

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary


class LLama(LLMModel):
    def __init__(self, device):
        self.model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        super().__init__(device)


    def generate(self, inpt):
        inputs = self.tokenizer(
            [inpt], max_length=1024, return_tensors="pt", truncation=True
        )
        summary_ids = self.model.generate(inputs["input_ids"])

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary


class Opt(LLMModel):
    def __init__(self, device):
        self.model_name = "facebook/opt-6.7b"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
            device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        super().__init__(device)
 
 
    def generate(self, inpt):
        inputs = self.tokenizer(
            [inpt], max_length=self.getMaxLength(), return_tensors="pt", truncation=True
        )
        summary_ids = self.model.generate(inputs["input_ids"])
 
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
 
        return summary


class Gptj(LLMModel):
    def __init__(self, device):
        self.model_name = "EleutherAI/gpt-j-6B"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
            device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        super().__init__(device)
 
 
    def generate(self, inpt):
        inputs = self.tokenizer(
            [inpt], max_length=self.getMaxLength(), return_tensors="pt", truncation=True
        )
        summary_ids = self.model.generate(inputs["input_ids"])
 
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
 
        return summary


class BartCNN(LLMModel):
    def __init__(self, device):
        self.model_name = "facebook/bart-large-cnn"
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name).to(device)
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        super().__init__(device)


    def generate(self, inpt):
        inputs = self.tokenizer(
            [inpt], max_length=self.getMaxLength(), return_tensors="pt", truncation=True
        )
        summary_ids = self.model.generate(inputs["input_ids"])

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary

class GPT2Generator(LLMModel):
    def __init__(self, device):
        self.model_name = "gpt2-large"
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name,
            device_map='auto')
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        super().__init__(device)

    def generate(self, inpt):
        inputs = self.tokenizer.encode(
            inpt, return_tensors="pt", max_length=self.getMaxLength(), truncation=True
        )
        output_ids = self.model.generate(
            inputs, max_length=1024, num_return_sequences=1
        )
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output


class LED(LLMModel):
    def __init__(self, device):
        self.model_name = "allenai/led-large-16384-arxiv"
        self.model = LEDForConditionalGeneration.from_pretrained(self.model_name).to(device)
        self.tokenizer = LEDTokenizer.from_pretrained(self.model_name)
        super().__init__(device)

    def generate(self, inpt):
        inputs = self.tokenizer(
            [inpt], max_length=self.getMaxLength(), return_tensors="pt", truncation=True
        )
        summary_ids = self.model.generate(inputs["input_ids"])

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary


class SimpleDenseNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim=1, dropout_prob=0.3):
        super(SimpleDenseNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
####################
## INitialization ##
####################

## Path creation
base_path =  Path(".")
output_path = base_path / "HalluDetect"
data_path = base_path
output_path.mkdir(exist_ok=True, parents=True)

def adaptDataset(
    train_data: pd.DataFrame, test_data: pd.DataFrame, includeConditioned: bool
):

    dataset_train = []
    dataset_test = []
    for _, row in train_data.iterrows():
        prompt, text, hallu = row["Prompt"], row["Result"], row["Label"]
        dataset_train.append((prompt, text, hallu))

    for _, row in test_data.iterrows():
        prompt, text, hallu = row["Prompt"], row["Result"], row["Label"]
        dataset_test.append((prompt, text, hallu))

    random.shuffle(dataset_train)
    random.shuffle(dataset_test)

    dataset_train = dataset_train[:2500]

    X_train = [(p if includeConditioned else "", t) for p, t, _ in dataset_train]
    Y_train = [y for _, _, y, in dataset_train]

    X_test = [(p if includeConditioned else "", t) for p, t, _ in dataset_test]
    Y_test = [y for _, _, y, in dataset_test]

    return X_train, Y_train, [], [], X_test, Y_test


def getXY(df: pd.DataFrame, includeKnowledge=True, includeConditioned=True):
    X = []
    Y = []

    # Iterate over rows using itertuples
    for _, row in df.iterrows():
        x, c, g = (
            row["Knowledge"] if includeKnowledge else "",
            row["Conditioned Text"] if includeConditioned else "",
            row["Generated Text"],
        )
        y = row["Label"]

        # Append values to respective lists
        X.append((x, c, g))
        Y.append(y)
    return X, Y


def extract_features(
    knowledge: str,
    conditioned_text: str,
    generated_text: str,
    features_to_extract: dict[str, bool],
    model
):
    return model.extractFeatures(
        knowledge, conditioned_text, generated_text, features_to_extract
    )

def compute_metrics(model, input_tensor, true_labels):
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_probs = torch.sigmoid(outputs).cpu().numpy()
        predicted = (outputs > 0.5).float().cpu().numpy()

        true_labels = true_labels.cpu().numpy()

        acc = accuracy_score(true_labels, predicted)
        precision = precision_score(true_labels, predicted)
        recall = recall_score(true_labels, predicted)
        f1 = f1_score(true_labels, predicted)

        precision_negative = precision_score(true_labels, predicted, pos_label=0)
        recall_negative = recall_score(true_labels, predicted, pos_label=0)
        f1_negative = f1_score(true_labels, predicted, pos_label=0)

        tn, fp, fn, tp = confusion_matrix(true_labels, predicted).ravel()
        roc_auc = roc_auc_score(true_labels, predicted_probs)

        P, R, thre = precision_recall_curve(true_labels, predicted, pos_label=1)
        pr_auc = auc(R, P)

        roc_auc_negative = roc_auc_score(
            true_labels, 1 - predicted_probs
        )  # If predicted_probs is the probability of the positive class
        P_neg, R_neg, _ = precision_recall_curve(true_labels, predicted, pos_label=0)
        pr_auc_negative = auc(R_neg, P_neg)

        return {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "ROC AUC": roc_auc,
            "PR AUC": pr_auc,
            "Precision-Negative": precision_negative,
            "Recall-Negative": recall_negative,
            "F1-Negative": f1_negative,
            "ROC AUC-Negative": roc_auc_negative,
            "PR AUC-Negative": pr_auc_negative,
            "Output": predicted.tolist(),
        }

def compute_accuracy(model, input_tensor, true_labels):
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted = (outputs > 0.5).float()
        correct = (predicted == true_labels).float().sum()
        accuracy = correct / len(true_labels)
        return accuracy.item()
    
def load_dataset(data, source_info):
    prompt_list = []
    result_list = []
    label_list = []
    for element in data:
        source_id = element["source_id"]
        label = 0 if len(element["labels"]) > 0 else 1
        response = element["response"]
        for source in source_info:
            if source["source_id"] == source_id:
                prompt = source["prompt"]
        prompt_list.append(prompt)
        result_list.append(response)
        label_list.append(label)

    test_data = pd.DataFrame(
        {"Prompt": prompt_list, "Result": result_list, "Label": label_list}
    )
    return test_data

def main(model, device):
    ## Load the dataset

    with open("dataset/source_info.json", "r")as f:
        source_info = json.load(f)

    with open("dataset/response.json", "r") as f:
        data = json.load(f)
    data = load_dataset(data, source_info)
    with open("dataset/training_data.json", "r") as f:
        training_data = []
        for line in f.readlines():
            training_data.append(json.loads(line.strip()))
    training_data = load_dataset(training_data, source_info)

    ## Features to extract
    feature_to_extract = 'all'

    available_features_to_extract = ["mtp", "avgtp", "MDVTP", "MMDVP"]
    if feature_to_extract == 'all':
        features_to_extract = {
            feature: True for feature in available_features_to_extract
        }
    else:
        features_to_extract = {
            feature: True if feature == feature_to_extract else False
            for feature in available_features_to_extract
        }

    # clean GPU
    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated(0), torch.cuda.memory_allocated(1))

    includeConditioned = True

    X_train, Y_train, _, _, X_test, Y_test = adaptDataset(
        training_data, data, includeConditioned
    )

    X_train_features_maps = []

    for conditioned_text, generated_text in tqdm(X_train, desc="Processing"):
        X_train_features_maps.append(
            extract_features(
                "", conditioned_text, generated_text, features_to_extract, model
            )
        )
        torch.cuda.empty_cache()  # Clean cache in every step for memory saving.
    print(torch.cuda.memory_allocated(0), torch.cuda.memory_allocated(1))  # Clean cache in every step for memory saving.

    X_train_features = [list(dic.values()) for dic in X_train_features_maps]

    clf = LogisticRegression(verbose=1)
    clf.fit(X_train_features, Y_train)

    X_test_features_map = []

    for conditioned_text, generated_text in tqdm(X_test, desc="Processing"):
        X_test_features_map.append(
            extract_features(
                "", conditioned_text, generated_text, features_to_extract, model
            )
        )
        torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated(0), torch.cuda.memory_allocated(1))

    X_test_features = [list(dic.values()) for dic in X_test_features_map]

    Y_Pred = clf.predict(X_test_features)

    lr_accuracy = accuracy_score(Y_test, Y_Pred)
    print(f"Accuracy: {lr_accuracy * 100:.2f}%")

    log_odds = clf.coef_[0]
    pd.DataFrame(log_odds, X_train_features_maps[0].keys(), columns=["coef"]).sort_values(
        by="coef", ascending=False
    )

    odds = np.exp(clf.coef_[0])
    pd.DataFrame(odds, X_train_features_maps[0].keys(), columns=["coef"]).sort_values(
        by="coef", ascending=False
    )

    del clf
    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated(0), torch.cuda.memory_allocated(1))

    denseModel = SimpleDenseNet(input_dim=len(list(features_to_extract.keys())), hidden_dim=512).to(device)

    X_train_tensor = torch.tensor(X_train_features, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(denseModel.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10000
    for epoch in range(num_epochs):
        denseModel.train()
        optimizer.zero_grad()
        outputs = denseModel(X_train_tensor)
        loss = criterion(outputs, Y_train_tensor)
        loss.backward()
        optimizer.step()

        # Compute training accuracy
        train_accuracy = compute_accuracy(denseModel, X_train_tensor, Y_train_tensor)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy:.4f}"
            )  # , "Validation Accuracy": {val_accuracy:.4f}')
    
    X_test_tensor = torch.tensor(X_test_features, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1).to(device)

    test_metrics = compute_metrics(denseModel, X_test_tensor, Y_test_tensor)

    print(
        f"Testing - Accuracy: {test_metrics['Accuracy']:.4f}, Precision: {test_metrics['Precision']:.4f}, Recall: {test_metrics['Recall']:.4f}, F1: {test_metrics['F1']:.4f}, ROC AUC: {test_metrics['ROC AUC']:.4f}, PR AUC: {test_metrics['PR AUC']:.4f}"
    )
    print(
        f"Testing - Negative: {test_metrics['Accuracy']:.4f}, Precision-Negative: {test_metrics['Precision-Negative']:.4f}, Recall-Negative: {test_metrics['Recall-Negative']:.4f}, F1-Negative: {test_metrics['F1-Negative']:.4f}, ROC AUC-Negative: {test_metrics['ROC AUC-Negative']:.4f}, PR AUC-Negative: {test_metrics['PR AUC-Negative']:.4f}"
    )

    d = {
        "features": features_to_extract,
        "model_name": str(model.getName()),
        "feature_to_extract": feature_to_extract,
        "method": "TEST",
        "accuracy": test_metrics["Accuracy"],
        "precision": test_metrics["Precision"],
        "recall": test_metrics["Recall"],
        "f1": test_metrics["F1"],
        "pr auc": test_metrics["PR AUC"],
        "precision-negative": test_metrics["Precision-Negative"],
        "recall-negative": test_metrics["Recall-Negative"],
        "negative-f1": test_metrics["F1-Negative"],
        "output": test_metrics["Output"]
    }
    
    name = f"{model.getSanitizedName()}{'_'.join([f'{k}={v}' for k, v in features_to_extract.items()])}.json"
    with open(output_path / name, "w") as f:
        json.dump(d, f, indent=4)


    del X_test_tensor
    del Y_test_tensor
    del X_test_features
    del X_train_features
    del X_train
    del X_test
    del Y_test
    del Y_train
    del Y_train_tensor
    del X_train_tensor
    del Y_Pred

if __name__ == '__main__':
    ## Setting device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    ## It is recommended to use 2 A100 GPUs or an equivalent amount of VRAM.

    models = [BartCNN, LED, GPT2Generator, LLama, Gemma, Opt, Gptj]
    for i in range(len(models)):
        model = models[i](device)
        main(model, device)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated(0), torch.cuda.memory_allocated(1))