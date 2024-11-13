# Copyright (c) 2024 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Lorenzo Paleari

import json
from collections import defaultdict

def calculate_accuracy_percentage(accurate: float, total: int) -> float:
    """
    Function to calculate accuracy percentage.

    :param accurate: Combined value of scores for the dataset.
    :type accurate: float
    :param total: Number of items in the dataset.
    :type total: int
    :return: Accuracy percentage.
    :rtype: float
    """
    if total == 0:
        return 0
    return (accurate / total) * 100


# Load the dataset
with open("dataset.json", "r") as f:
    dataset = json.load(f)

# Initialize the new dataset
categorized_dataset = defaultdict(list)

# Iterate over the dataset
for passage_number, value in dataset.items():
    annotation = value["annotation"]
    
    # Count the number of accurate annotations
    accurate = sum(1 for label in annotation if label == "accurate")
    half_accurate = sum(0.5 for label in annotation if "minor" in label)
    
    # Calculate total annotations
    total_annotations = len(annotation)
    
    # Calculate accuracy percentage
    accuracy_percentage = calculate_accuracy_percentage(accurate + half_accurate, total_annotations)
    
    # Add the passage number to the respective category in the new dataset
    categorized_dataset[passage_number] = accuracy_percentage

with open("./passage_scores.json", "w") as outfile:
    json.dump(categorized_dataset, outfile, indent=4)
