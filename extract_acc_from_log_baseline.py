import json
import re
import os
import glob
def extract_results_from_log(log_file, output_file_path=None):

    with open(log_file + os.sep +  "eval.log", 'r') as file:
        content = file.read()
    
    # Use regex to find the results dictionary
    pattern = r"(\{'Overall-Art and Design':.+?'Overall': \{'num': \d+, 'acc': \d+\.\d+\}\})"
    match = re.search(pattern, content, re.DOTALL)
    
    # Convert the string representation to an actual dictionary
    results_str = match.group(1)
    results = eval(results_str)
    
    # Save to JSON if output path is provided
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    with open(output_file_path + os.sep + "eval.json", "w") as outfile:
        json.dump(results, outfile, indent=4)
    print(f"Results saved to {output_file_path}")

    return results



eval_task_names = ["mmmu", "mme"]  # "mmlu" # "mmmu"
task_name = "entity-visual"
model_paths = ["Qwen/Qwen2.5-VL-7B-Instruct"]


futures = []
for eval_task_name in eval_task_names:
    for model_path in model_paths:

            model_name = os.path.basename(model_path)
            log_file= f"/home/winnieyangwn/lmms-eval/LOGS/{model_path}/{eval_task_name}"
            output_file = f"/home/winnieyangwn/Output/GENERAL/{eval_task_name}/{model_name}/"
            extract_results_from_log(log_file, output_file)


