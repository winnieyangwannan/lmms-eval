import re
import numpy as np
import os
import json
def extract_celebrity_result(log_file_path):
    """
    Extract the celebrity aggregate result from the evaluation log file.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        float: The celebrity score, or None if not found
    """
    try:
        with open(log_file_path, 'r') as file:
            content = file.read()
            
        # Pattern to match the celebrity line
        # Looking for: [INFO] | utils:mme_aggregate_results:124 - celebrity: 69.71
        pattern = r'utils.*mme_aggregate_results.*celebrity:\s*(\d+\.?\d*)'
        
        match = re.search(pattern, content)
        if match:
            return float(match.group(1))
        else:
            return None
            
    except FileNotFoundError:
        print(f"File not found: {log_file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    

def save_results_to_file(results, output_file_path):
    """
    Save the extracted results to a file.

    Args:
        results (float): The extracted celebrity score
        output_file_path (str): Path to save the results
    """

    results = {"celebrity_score": results}
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    
    with open(output_file_path + os.sep + "eval.json", 'w') as outfile:
        json.dump(results, outfile, indent=4)
    print(f"Results saved to {output_file_path}")    

# log_file_path = "/home/winnieyangwn/lmms-eval/LOGS/entity-visual/Qwen2.5-VL-7B-Instruct/positive-negative-addition-same/last/layer_0/strength_1/mlp-down/epoch_99/mme/eval.log"

# acc = extract_celebrity_result(log_file_path)

# print(acc)

eval_task_name = "mme"  # "mme" # "mmmu"
task_name = "entity-visual"
model_name_ogs = ["Qwen2.5-VL-7B-Instruct"]
# steer_types = ["negative-addition"]
# steer_types = ["positive-negative-addition-opposite"]
steer_types = ["positive-negative-addition-same"]
return_type = "prompt"
train_module = "mlp-down" #block
steer_poses = ["last"]  # "entity"
steering_strengths =[1, 2,3,4]
# entity_types = ["song"]
# entity_types = ["all"]
# entity_types = ["player", "city", "movie", "song", "all"]
# steer_poses = ["last", "entity"]  # "entity"
# known_unknown_split = "3"
epoch = 99


futures = []
for steering_strength in steering_strengths:
    for steer_pos in steer_poses:
        for model_name_og in model_name_ogs:
            for steer_type in steer_types:
                    if  "Qwen2.5-VL-3B" in model_name_og:
                                layers = np.arange(0, 35, 2)
                                layers = [15]
                    elif  "Qwen2.5-VL-7B" in model_name_og:
                                layers = np.arange(0, 27, 2)
                    elif  "gemma-2-9b-it" in model_name_og:
                        layers = np.arange(0, 42, 2)
                        # layers = [40]
                    elif "Llama-3.1-8B-Instruct" in model_name_og:
                        layers = np.arange(0, 32+2, 2)
                        # layers = [18]
                        # layers = [22]
                    elif "Qwen3-30B-A3B" in model_name_og:
                        layers = np.arange(0, 47, 2)
                        # layers = [22]
                    for layer in layers:
                            model_path = f"{task_name}_{model_name_og}_{train_module}_{steer_type}_{steer_pos}_layer_{layer}_{steering_strength}_{epoch}"
                            huggingface_path= "winnieyangwannan/" + model_path
                            # save_path= f"/home/winnieyangwn/Output/entity_training/{return_type}/{model_name_og}/{steer_type}/{steer_pos}/layer_{layer}/strength_{steering_strength}/{entity_type}/{known_unknown_split}/{train_module}/epoch_49"
                            log_file= f"/home/winnieyangwn/lmms-eval/LOGS/{task_name}/{model_name_og}/{steer_type}/{steer_pos}/layer_{layer}/strength_{steering_strength}/{train_module}/epoch_{epoch}/{eval_task_name}"
                            output_file = f"/home/winnieyangwn/Output/entity_visual_training/{return_type}/{model_name_og}/{steer_type}/{steer_pos}/layer_{layer}/strength_{steering_strength}/{train_module}/epoch_{epoch}/{eval_task_name}"
                            results = extract_celebrity_result(log_file)
                            save_results_to_file(results, output_file)