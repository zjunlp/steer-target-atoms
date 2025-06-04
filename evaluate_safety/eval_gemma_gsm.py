import json
import re
import argparse
import os
import csv
def load_data(file_path: str) -> list:
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def rewrite_json(path, data):
    with open(path, 'a') as file:
        json.dump(data, file, indent=4)
        file.write(',')
        file.write('\n')
        file.flush()
def rewrite_csv(path, data):
    file_exists = os.path.exists(path)
    with open(path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def process_predictions(pred_file: str, output_file: str=None):
    data = load_data(pred_file)

    correct_count = 0
    total_count = len(data)
    no_match_count = 0
    multiple_match_count = 0
    no_match_indices = []
    multiple_match_indices = []

    for index, entry in enumerate(data):
        # matches = re.findall(r'\$\\boxed{(\d+)}\$', entry['pred'])
        # matches = re.findall(r'\$\\boxed{([^}]+)}\$', entry["pred"])
        # matches = re.findall(r'\$\\boxed{(.*?)}\$', entry["pred"])
        pred = entry["pred"].split('\n\nQuestion:')[0]
        matches = re.findall(r'\$\\boxed{(.*?)}', pred)
        
        if len(matches) == 0:
            no_match_count += 1
            no_match_indices.append(f'{index}-{data[index]["pred"]}')
            pred_answer = None
        elif len(matches) == 1:
            pred_answer = matches[0]
        else:
            multiple_match_count += 1
            multiple_match_indices.append(index)
            pred_answer = matches[-1]

        actual_answer = entry['answer']

        if pred_answer is not None and pred_answer in actual_answer:
            correct_count += 1

    accuracy = correct_count / total_count
    no_match_ratio = no_match_count / total_count
    multiple_match_ratio = multiple_match_count / total_count

    results = {
        "pred_file": os.path.basename(pred_file),
        "accuracy": f"{accuracy * 100:.2f}%",
        "no_match_ratio": f"{no_match_ratio * 100:.2f}%",
        "multiple_match_ratio": f"{multiple_match_ratio * 100:.2f}%",
        "no_match_indices": no_match_indices,
        "multiple_match_indices": multiple_match_indices
    }

    rewrite_json(f'{pred_file}_eval_result.json', results)
    # rewrite_json(f'{output_file}/gemma_gsm.json', results)
    # rewrite_csv(f'{output_file}/gemma_gsm.csv', results)

    print(f'pred_file: {pred_file}\n\n results:{results}')

def find_pred_files(path):
    pred_files = []
    if os.path.isfile(path):
        pred_files.append(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.result.json'):
                    pred_files.append(os.path.join(root, file))
    return pred_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, help="Path to the input prediction file (JSON).")
    # parser.add_argument("--output_file", type=str, help="Path to the output file for saving results (JSON).")
    args = parser.parse_args()

    pred_files = find_pred_files(args.pred_file)
    if not pred_files:
        print(f"Error: No .json files found (input path: {args.pred_file})")
    
    for pred_file in pred_files:
        process_predictions(pred_file)
        # process_predictions(args.pred_file, args.output_file)
