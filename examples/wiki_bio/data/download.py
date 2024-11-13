# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

from datasets import load_dataset
import json

ds = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
ds = ds["evaluation"]

features = ['gpt3_text', 'wiki_bio_text', 'gpt3_sentences', 'annotation', 'wiki_bio_test_idx', 'gpt3_text_samples']
dataset = {}
for feat in features:
    dataset.update({feat: ds[feat]})

dataset_final = {}
for i in range(len(dataset[features[0]])):
    dataset_passage = {}
    for feat in features:
        dataset_passage.update({feat: dataset[feat][i]})
    name = f"passage_{i}"
    dataset_final.append({
        name: dataset_passage
    })

with open("dataset.json", "w") as f:
    json.dump(dataset_final, f, indent=4)
