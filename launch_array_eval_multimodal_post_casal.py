import os
import concurrent.futures
import subprocess
import re
from os.path import join
import argparse
import numpy as np
# model_path = "meta-llama/Llama-3.1-8B"



def run_subprocess_slurm(command):
    # Execute the sbatch command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Print the output and error, if any
    print("command:", command)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    # Get the job ID for linking dependencies
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = match.group(1)
    else:
        job_id = None

    return job_id



current_directory = os.getcwd()
print("current_directory:", current_directory)
eval_task_name = "mmmu"
gpus_per_node = 1
task_name = "entity-visual"
model_name_ogs = ["Qwen2.5-VL-7B-Instruct"]
# steer_types = ["negative-addition"]
# steer_types = ["positive-negative-addition-opposite"]
steer_types = ["positive-negative-addition-same"]
return_type = "prompt"
train_module = "mlp-down" #block
steer_poses = ["last"]  # "entity"
steering_strength =1
# entity_types = ["all"]
# entity_types = ["player", "city", "movie", "song", "all"]
# steer_poses = ["last", "entity"]  # "entity"
# known_unknown_split = "3"
epoch = 99


# Run the SLURM commands in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:

    futures = []
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
                            job_name  = f"eval"
                            save_path= f"LOGS/{task_name}/{return_type}/{model_name_og}/{steer_type}/{steer_pos}/layer_{layer}/strength_{steering_strength}/{train_module}/epoch_{epoch}/{eval_task_name}"

                            slurm_cmd = f'''sbatch --account=genai_interns --qos=lowest \
                                --job-name={job_name} --nodes=1 --gpus-per-node={gpus_per_node} \
                                --time=24:00:00 --output={save_path}/eval.log \
                                --wrap="\
                                    accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
                                    --model qwen2_5_vl \
                                    --model_args=pretrained={huggingface_path},max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=True \
                                    --tasks mmmu_val ; \

                            "'''
                            
                            job_id = run_subprocess_slurm(slurm_cmd)
                            print("job_id:", job_id)

# os.chdir(current_directory + os.sep + "src" + os.sep + "eval") 
# os.chdir(current_directory + os.sep + "SUBMODULES/GENERAL_CAPABILITY") 
# model_name = "meta-llama/Llama-3.1-8B"  # winnieyangwannan/refusal_Llama-3.1-8B-Instruct_mlp_positive-negative-addition-opposite_last_layer_18_2_49
# model_name = "Qwen2.5-VL-7B"
# if  "Qwen2.5-VL-3B" in model_name:
#     layers = np.arange(0, 35, 2)
#     layers = [15]
# elif  "Qwen2.5-VL-7B" in model_name:
#     layers = np.arange(0, 27, 2)
# elif "Llama-3.2-11B-Vision-Instruct" in model_name:
#     layers = np.arange(0, 39+2, 2)
# task_name = "mmmu" # "mmlu"

# for layer in layers:
#     model_path= f"winnieyangwannan/entity-visual_Qwen2.5-VL-7B-Instruct_mlp-down_negative-addition_last_layer_{layer}_1_49" # #"Qwen/Qwen2.5-VL-7B-Instruct"
#     model_name = os.path.basename(model_path)  # Extract the model name from the path

#     job_name  = f"generate_{task_name}_{model_name}"
#     slurm_cmd = f'''sbatch --account=genai_interns --qos=lowest \
#         --job-name={job_name} --nodes=1 --gpus-per-node={gpus_per_node} \
#         --time=24:00:00 --output=LOGS/{job_name}.log \
#         --wrap="\
#             accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#             --model qwen2_5_vl \
#             --model_args=pretrained={model_path},max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=True \
#             --tasks mmmu_val \
#             --batch_size 1 \
#             --log_samples \
#             --log_samples_suffix reproduce \
#             --output_path /home/winnieyangwn/Output/GENERAL/ ; \
#     "'''
    
#     job_id = run_subprocess_slurm(slurm_cmd)
#     print("job_id:", job_id)


