import torch
from torchvision.datasets import EMNIST
from collections import Counter
import numpy as np
import os
import datetime

def analyze_emnist_distribution(split_name: str, output_file):
    """
    Analyzes and writes the class distribution for a given EMNIST split to a file.

    Args:
        split_name (str): The name of the EMNIST split (e.g., 'byclass', 'bymerge').
        output_file: A file handle to write the output to.
    """
    output_file.write(f"\n{'='*25} Analyzing EMNIST/{split_name.upper()} {'='*25}\n")

    # Define the data directory
    data_dir = './data'
    
    # Load the training dataset
    try:
        # We don't need transforms as we only need the labels
        print(f"Loading EMNIST/{split_name} dataset... (This may take a moment)")
        dataset = EMNIST(root=data_dir, split=split_name, train=True, download=True)
        output_file.write("Dataset loaded successfully.\n")
    except Exception as e:
        error_msg = f"Error: Could not download or load the dataset from '{data_dir}'.\nPlease check your internet connection or directory permissions.\nDetails: {e}\n"
        output_file.write(error_msg)
        print(error_msg) # Also print critical errors to console
        return

    # Get labels and class-to-character mapping
    labels = dataset.targets
    # The `classes` attribute gives the readable character names
    class_to_char_mapping = dataset.classes

    # Count the occurrences of each class label (which are integers)
    class_counts = Counter(labels.numpy())

    # Create a list of tuples: (character, count, original_class_index)
    distribution_data = []
    for class_idx, count in class_counts.items():
        character = class_to_char_mapping[class_idx]
        distribution_data.append((character, count, class_idx))

    # Sort the data by count (ascending)
    sorted_distribution = sorted(distribution_data, key=lambda x: x[1])

    # Write the results in a formatted table to the file
    output_file.write("-" * 65 + "\n")
    output_file.write(f"{'Rank':<5} | {'Character':<10} | {'Class Index':<12} | {'Sample Count':<15}\n")
    output_file.write(f"{'-'*5}-+-{'-'*10}-+-{'-'*12}-+-{'-'*15}\n")

    for rank, (character, count, class_idx) in enumerate(sorted_distribution, 1):
        output_file.write(f"{rank:<5} | {character:<10} | {class_idx:<12} | {f'{count:,}':<15}\n")
    output_file.write("-" * 65 + "\n")

    # --- Suggest a threshold for minority classes ---
    counts = np.array([count for _, count, _ in sorted_distribution])
    
    # A simple but effective heuristic for a threshold: 50% of the median count.
    # This identifies classes that are significantly smaller than the "typical" class.
    median_count = np.median(counts)
    suggested_threshold = median_count * 0.5 
    
    output_file.write("\n--- Minority Class Analysis & Threshold ---\n")
    output_file.write(f"Total number of classes: {len(counts)}\n")
    output_file.write(f"Median class size: {int(median_count):,}\n")
    output_file.write(f"Average class size: {int(np.mean(counts)):,}\n")
    output_file.write(f"\nSuggested Threshold for 'Minority Class': < {int(suggested_threshold):,} samples (50% of median)\n")

    minority_classes = [d for d in sorted_distribution if d[1] < suggested_threshold]
    
    if not minority_classes:
        output_file.write("No classes fall below this threshold. The dataset is relatively balanced.\n")
    else:
        output_file.write(f"Found {len(minority_classes)} classes below this threshold. They are:\n")
        minority_chars = [f"'{char}' ({count:,} samples)" for char, count, _ in minority_classes]
        # Print in columns for better readability
        for i in range(0, len(minority_chars), 4):
             output_file.write("   ".join(minority_chars[i:i+4]) + "\n")


if __name__ == "__main__":
    # Ensure the data directory exists
    os.makedirs('./data', exist_ok=True)
    
    # Define log file path
    log_file_path = 'emnist_distribution_analysis.log'
    
    print(f"Starting EMNIST distribution analysis. Results will be saved to '{log_file_path}'...")
    
    # Open the file and run the analysis
    with open(log_file_path, 'w') as f:
        f.write(f"EMNIST Dataset Distribution Analysis\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        analyze_emnist_distribution('byclass', f)
        analyze_emnist_distribution('bymerge', f)
        
    print(f"Analysis complete.") 