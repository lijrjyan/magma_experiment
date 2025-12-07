import re
import ast
import os
import argparse

def parse_log_for_rounds(log_content, round_numbers):
    """
    Parses the log file content to extract information for specified rounds.

    Args:
        log_content (str): The entire content of the log file.
        round_numbers (list[int]): A list of round numbers to parse.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains
                    the extracted information for a single round.
    """
    results = []
    
    # Find all round starts to define round blocks
    round_starts = {}
    # Use a wide range for rounds to be safe
    for r in range(1, 500):
        # Use word boundary \b to ensure matching "round 3" but not "round 30"
        pattern = re.compile(rf"start training round {r}\b")
        # Use finditer to get all matches if a round is restarted, take the last one
        matches = list(pattern.finditer(log_content))
        if matches:
            round_starts[r] = matches[-1].start()

    sorted_rounds = sorted(round_starts.keys())

    for round_number in round_numbers:
        if round_number not in round_starts:
            continue

        start_index = round_starts[round_number]
        
        # Find the start of the next round to define the end of the current block
        try:
            next_round_index_in_list = sorted_rounds.index(round_number) + 1
            if next_round_index_in_list < len(sorted_rounds):
                next_round_start_key = sorted_rounds[next_round_index_in_list]
                end_index = round_starts[next_round_start_key]
            else:
                end_index = len(log_content)
        except (ValueError, IndexError):
             end_index = len(log_content)
        
        round_log = log_content[start_index:end_index]

        # 1. First client ID
        data_sizes_match = re.search(r"data sizes for aggregation: ({.*?})", round_log)
        if not data_sizes_match:
            continue
        
        try:
            data_sizes_dict = ast.literal_eval(data_sizes_match.group(1))
            first_client_id = list(data_sizes_dict.keys())[0]
        except (ValueError, SyntaxError):
            continue

        # 2. Norm Scores
        norm_scores_match = re.search(rf"client {first_client_id} norm scores: ({{.*?}})", round_log, re.DOTALL)
        if not norm_scores_match:
            continue
        try:
            norm_scores_str_raw = norm_scores_match.group(1).replace('\n', '').replace(' ', '')
            norm_scores_dict = ast.literal_eval(norm_scores_str_raw)
            norm_scores_str = ", ".join([f"{k}: {v:.2f}" for k, v in norm_scores_dict.items()])
        except (ValueError, SyntaxError):
            norm_scores_str = "Error parsing"

        # 3. Threshold
        threshold_match = re.search(rf"client {first_client_id} similarity threshold: ([\d.-]+)", round_log)
        if not threshold_match:
            continue
        try:
            threshold = float(threshold_match.group(1))
            threshold_str = f"{threshold:.2f}"
        except ValueError:
            threshold_str = "Error parsing"

        # 4. Selected Benigns
        selected_benigns_match = re.search(rf"client {first_client_id} selected fusion benigns: (\[.*?\])", round_log)
        if not selected_benigns_match:
            continue
        selected_benigns = selected_benigns_match.group(1)

        # 5. Accuracy
        accuracy_match = re.search(r"global side -  test accuracy: ([\d.]+)", round_log)
        if not accuracy_match:
            continue
        try:
            accuracy = float(accuracy_match.group(1))
            accuracy_str = f"{accuracy:.2f}"
        except ValueError:
            accuracy_str = "Error parsing"

        results.append({
            "Round": round_number,
            "Norm Scores": norm_scores_str,
            "Threshold": threshold_str,
            "Selected Benigns": selected_benigns,
            "Acc After the Round": accuracy_str
        })
        
    return results

def generate_markdown_table(results):
    """Generates a markdown table from the parsed results."""
    if not results:
        return "No data found for the specified rounds."
        
    header = "| Round | Norm Scores | Threshold | Selected Benigns | Acc After the Round |"
    separator = "|-------|-------------|-----------|------------------|---------------------|"
    
    lines = [header, separator]
    
    for res in results:
        lines.append(f"| {res['Round']} | {res['Norm Scores']} | {res['Threshold']} | {res['Selected Benigns']} | {res['Acc After the Round']} |")
        
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Parse federated learning log files.")
    parser.add_argument("log_file", type=str, help="Path to the log file.")
    parser.add_argument("rounds", type=int, nargs='+', help="List of round numbers to parse.")
    args = parser.parse_args()

    log_file_path = args.log_file
    target_rounds = args.rounds
    
    try:
        with open(log_file_path, 'r') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return

    parsed_data = parse_log_for_rounds(log_content, target_rounds)
    
    markdown_output = generate_markdown_table(parsed_data)
    
    print(f"--- Parsed rounds {target_rounds} from {os.path.basename(log_file_path)} ---")
    print(markdown_output)

    # Create the dynamic output path
    log_filename_no_ext = os.path.splitext(os.path.basename(log_file_path))[0]
    output_dir = os.path.join('Tables', log_filename_no_ext)
    os.makedirs(output_dir, exist_ok=True)

    rounds_str = "_".join(map(str, target_rounds))
    output_filename = f"Round{rounds_str}.md"
    output_filepath = os.path.join(output_dir, output_filename)

    with open(output_filepath, 'w') as f:
        f.write(markdown_output)
    
    print(f"\nTable also saved to {output_filepath}\n")


if __name__ == "__main__":
    main() 