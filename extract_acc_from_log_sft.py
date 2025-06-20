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



eval_task_name = "mmmu"  # "mmlu" # "mmmu"
task_name = "entity-visual"
model_name_ogs = ["Qwen2.5-VL-7B-Instruct"]
# steer_types = ["positive-negative-addition-opposite"]
# steer_types = ["positive-negative-addition-same"]
return_type = "prompt"
steer_poses = ["last"]  # "entity"
# entity_types = ["all"]
# entity_types = ["player", "city", "movie", "song", "all"]
# steer_poses = ["last", "entity"]  # "entity"


futures = []
for steer_pos in steer_poses:
    for model_name_og in model_name_ogs:

            model_path = f"{task_name}_{model_name_og}_sft_lfw"
            huggingface_path= "winnieyangwannan/" + model_path
            # save_path= f"/home/winnieyangwn/Output/entity_training/{return_type}/{model_name_og}/{steer_type}/{steer_pos}/layer_{layer}/strength_{steering_strength}/{entity_type}/{known_unknown_split}/{train_module}/epoch_49"
            log_file= f"/home/winnieyangwn/LMMS-EVAL/LOGS/{task_name}/{model_name_og}/sft/{eval_task_name}"
            output_file = f"/home/winnieyangwn/Output/entity_visual_sft-training/"
            extract_results_from_log(log_file, output_file)
        
        # if results:
        #     all_results[layer_num] = results.get('Overall', {}).get('acc', 0)



# # Example usage
# model_name = "Qwen2.5-VL-7B-Instruct_mlp-down_negative-addition_last_layer_0_1_49"  # Example model name
# log_file = f"/home/winnieyangwn/lmms-eval/LOGS/generate_mmmu_entity-visual_{model_name}.log"
# output_file = f"/home/winnieyangwn/Output/GENERAL/{model_name}/mmmu"

# # Extract and save results
# results = extract_results_from_log(log_file, output_file)


