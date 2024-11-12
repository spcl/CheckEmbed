import json
from collections import defaultdict

# Load the dataset
with open("dataset.json", "r") as f:
    dataset = json.load(f)

# Initialize the new dataset
categorized_dataset = defaultdict(list)

# Function to calculate accuracy percentage
def calculate_accuracy_percentage(accurate, total):
    if total == 0:
        return 0
    return (accurate / total) * 100

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

# Print the newly categorized dataset
#print(json.dumps(categorized_dataset, indent=4))

with open("./passage_scores.json", "w") as outfile:
    json.dump(categorized_dataset, outfile, indent=4)
