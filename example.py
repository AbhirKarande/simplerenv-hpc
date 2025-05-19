import os
import subprocess
import zipfile
import simpler_env
import tensorflow as tf
import logging
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import json
import numpy as np
import sapien.core as sapien
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARNING, 3=no INFO/WARNING/ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"GPU memory growth configuration error: {e}")

# Set environment variables for CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

RT_1_CHECKPOINTS = {
    "rt_1_x": "rt_1_x_tf_trained_for_002272480_step",
    "rt_1_400k": "rt_1_tf_trained_for_000400120",
    "rt_1_58k": "rt_1_tf_trained_for_000058240",
    "rt_1_1k": "rt_1_tf_trained_for_000001120",
}

def download_from_gs(gs_path, local_path):
    """Download a folder or file from a GCS path to local path using gsutil."""
    try:
        subprocess.run(["gsutil", "-m", "cp", "-r", gs_path, local_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading from {gs_path}: {e}")
        raise

def unzip_file(zip_path, extract_to):
    """Unzip a file to a specific directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def get_rt_1_checkpoint(name, ckpt_dir="./SimplerEnv/checkpoints"):
    assert name in RT_1_CHECKPOINTS, f"Unknown checkpoint name: {name}"
    ckpt_name = RT_1_CHECKPOINTS[name]
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_dir, exist_ok=True)
        if name == "rt_1_x":
            zip_file_path = os.path.join(ckpt_dir, f"{ckpt_name}.zip")
            gs_path = f"gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/{ckpt_name}.zip"
            download_from_gs(gs_path, zip_file_path)
            unzip_file(zip_file_path, ckpt_dir)
        else:
            gs_path = f"gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/{ckpt_name}"
            download_from_gs(gs_path, ckpt_dir)

    return ckpt_path
    
def download_all_rt_1_checkpoints(ckpt_dir="./SimplerEnv/checkpoints"):
    for name in RT_1_CHECKPOINTS:
        print(f"Downloading checkpoint: {name}")
        path = get_rt_1_checkpoint(name, ckpt_dir)
        print(f"✅ Downloaded {name} to {path}\n")

# Call the function to download all checkpoints
download_all_rt_1_checkpoints()

# @title Select your model and environment

task_name = "google_robot_move_near"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]

if 'env' in locals():
  print("Closing existing env")
  env.close()
  del env
env = simpler_env.make(task_name)

# Note: we turned off the denoiser as the colab kernel will crash if it's turned on
# To use the denoiser, please git clone our SIMPLER environments
# and perform evaluations locally.
sapien.render_config.rt_use_denoiser = False

obs, reset_info = env.reset()
instruction = env.get_language_instruction()
print("Reset info", reset_info)
print("Instruction", instruction)

if "google" in task_name:
  policy_setup = "google_robot"
else:
  policy_setup = "widowx_bridge"
 
# @title Select your model and environment
model_name = "rt_1_x" # @param ["rt_1_x", "rt_1_400k", "rt_1_58k", "rt_1_1k", "octo-base", "octo-small"]

if "rt_1" in model_name:
  from simpler_env.policies.rt1.rt1_model import RT1Inference

  ckpt_path = get_rt_1_checkpoint(model_name)
  model = RT1Inference(saved_model_path=ckpt_path, policy_setup=policy_setup)
elif "octo" in model_name:
  from simpler_env.policies.octo.octo_model import OctoInference

  model = OctoInference(model_type=model_name, policy_setup=policy_setup, init_rng=0)
else:
  raise ValueError(model_name)


#@title Run inference



obs, reset_info = env.reset()
instruction = env.get_language_instruction()
model.reset(instruction)
print(instruction)

image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
images = [image]
predicted_terminated, success, truncated = False, False, False
actions = []
timestep = 0
while not (predicted_terminated or truncated):
    # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
    try:
        raw_action, action, action_info = model.step(image)
    except ValueError:
        # Handle case where model only returns 2 values
        raw_action, action = model.step(image)
        action_info = None
    predicted_terminated = bool(action["terminate_episode"][0] > 0)
    actions.append(action)
    # step the environment
    obs, reward, success, truncated, info = env.step(
        np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
    )
    print(timestep, info, raw_action, action, action_info)
    # update image observation
    #add time_step, observation,
    image = get_image_from_maniskill2_obs_dict(env, obs)
    images.append(image)
    timestep += 1

for action in actions:
  for key, value in action.items():
    if isinstance(value, np.ndarray):
      action[key] = value.tolist()




print(f"Episode success: {success}")

