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

task_name = "google_robot_close_drawer"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]

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


import json
import numpy as np
import os
import gc
from datetime import datetime, timedelta

# Create a folder for all JSON files
output_folder = "inference_results_terminated"
os.makedirs(output_folder, exist_ok=True)

# Initialize data structure to store minimal summary information
summary_data = []

# For logging progress
start_time = datetime.now()
last_log_time = start_time

def log_progress(current_ep, total_ep, success_count):
    """Log progress with time estimates"""
    global last_log_time
    current_time = datetime.now()
    elapsed = (current_time - start_time).total_seconds() / 60  # minutes
    time_per_ep = elapsed / (current_ep + 1)
    remaining = time_per_ep * (total_ep - current_ep - 1)

    # Only log every 5 minutes or when specifically requested
    if (current_time - last_log_time).total_seconds() >= 300 or current_ep == total_ep - 1:
        print(f"Progress: {current_ep+1}/{total_ep} episodes completed ({(current_ep+1)/total_ep*100:.1f}%)")
        print(f"Success rate: {success_count}/{current_ep+1} ({success_count/(current_ep+1)*100:.1f}%)")
        print(f"Elapsed time: {elapsed:.1f} minutes")
        print(f"Estimated remaining time: {remaining:.1f} minutes")
        print(f"Estimated completion: {current_time + timedelta(minutes=remaining)}")
        print("-" * 50)
        last_log_time = current_time

# Helper function to ensure all numpy arrays are converted to lists
def ensure_serializable(obj):
    """Convert any non-serializable objects to JSON-serializable ones"""
    # Handle JAX Array types (ArrayImpl)
    if str(type(obj).__name__) == 'ArrayImpl':
        # Convert JAX array to numpy array first, then to list
        try:
            return obj.tolist()
        except:
            # If tolist fails, try numpy conversion first
            try:
                return np.array(obj).tolist()
            except:
                # Last resort: convert to string
                return str(obj)

    # Handle numpy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    # Handle dictionaries
    elif isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}

    # Handle lists and tuples
    elif isinstance(obj, (list, tuple)):
        return [ensure_serializable(i) for i in obj]

    # Handle numpy scalar types
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)

    # Check if object has a tolist method (for other array-like objects)
    elif hasattr(obj, 'tolist'):
        try:
            return obj.tolist()
        except:
            return str(obj)

    # Return object as is if it's likely JSON serializable
    else:
        return obj

# A JSON encoder class that uses our serialization function
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return ensure_serializable(obj)
        except:
            return str(obj)  # Fallback to string representation

# Track success count for reporting
success_count = 0

# Run for 50 episodes
for episode_id in range(50):
    print(f"Running episode {episode_id+1}/50")

    # Reset environment for a new episode
    obs, reset_info = env.reset()
    instruction = env.get_language_instruction()
    model.reset(instruction)

    if episode_id == 0:  # Only print instruction for first episode to reduce output clutter
        print(f"Instruction: {instruction}")

    # Create an episode dictionary - we'll flush this after each episode
    episode_data = []

    # Track episode outcome
    success = False
    elapsed_steps = 0
    episode_stats = {"n_lift_significant": 0, "consec_grasp": False, "grasped": False}

    # Run 100 timesteps for the episode
    for timestep in range(100):
        if timestep % 20 == 0:  # Print even less frequently to reduce clutter
            print(f"  Timestep {timestep}")

        elapsed_steps += 1

        # Get the image (use single image now)
        image = get_image_from_maniskill2_obs_dict(env, obs) # Use single image

        # --- Perform a single inference ---
        try:
            # Pass the single image (H, W, 3)
            raw_action, action, action_info = model.step(image)
        except ValueError:
            # Handle models that might not return action_info
            raw_action, action = model.step(image)
            action_info = None

        # Cleanup the image used for inference
        del image

        # Store data for logging (convert to serializable format)
        serializable_raw_action = ensure_serializable(raw_action)
        serializable_action = ensure_serializable(action)

        # Extract optional info like entropy, log_probs, token_argmax for logging
        token_argmax_data = []
        token_entropy_data = [] # Renamed from token_entropy_by_inference for clarity
        if action_info:
            if 'entropy' in action_info:
                entropy_data = ensure_serializable(action_info['entropy'])
                if isinstance(entropy_data, list) and len(entropy_data) > 0:
                    token_entropy_data.append(entropy_data[0]) # Store first horizon entropy

            if 'token_argmax' in action_info:
                 token_argmax_val = ensure_serializable(action_info['token_argmax'])
                 if isinstance(token_argmax_val, list) and len(token_argmax_val) > 0:
                    token_argmax_data.append(token_argmax_val[0]) # Store first horizon argmax

        # --- Use the single processed action to step Environment ---
        # Ensure action components are numpy arrays before concatenation
        world_vector = np.array(action["world_vector"])
        rot_axangle = np.array(action["rot_axangle"])
        gripper = np.array(action["gripper"])

        combined_action = np.concatenate([world_vector, rot_axangle, gripper])
        obs, reward, terminated, truncated, info = env.step(combined_action)

        # --- Cleanup action components and combined action ---
        del world_vector, rot_axangle, gripper, combined_action
        # Raw action and processed action are needed for logging below
        # if raw_action: del raw_action
        # if action: del action
        if action_info: del action_info

        # Get info from environment (ensure they're Python native types)
        is_grasped = bool(info.get("is_grasped", False))
        consecutive_grasp = bool(info.get("consecutive_grasp", False))
        lifted_object = bool(info.get("lifted_object", False))
        lifted_object_significantly = bool(info.get("lifted_object_significantly", False))
        success = bool(info.get("success", False))

        # Update episode stats
        if lifted_object_significantly:
            episode_stats["n_lift_significant"] += 1
        if consecutive_grasp:
            episode_stats["consec_grasp"] = True
        if is_grasped:
            episode_stats["grasped"] = True

        # Store timestep data in the format shown in the JSON snippet
        timestep_data = {
            "timestep": timestep,
            # Store the single action components used to step the env
            "action_components": serializable_action,
            "raw_action": serializable_raw_action,   # Log the raw action
            "info": {
                "elapsed_steps": elapsed_steps,
                "is_grasped": is_grasped,
                "consecutive_grasp": consecutive_grasp,
                "lifted_object": lifted_object,
                "lifted_object_significantly": lifted_object_significantly,
                "success": success,
                "episode_stats": episode_stats
            }
        }

        if token_argmax_data:
            timestep_data["token_argmax"] = token_argmax_data

        # Add the token_entropy data if available
        if token_entropy_data:
            timestep_data["token_entropy"] = token_entropy_data

        episode_data.append(timestep_data)



    # Save this episode's data to a separate JSON file in the folder
    episode_file_path = os.path.join(output_folder, f'episode_{episode_id}_data.json')
    try:
        # First attempt with our enhanced encoder
        with open(episode_file_path, 'w') as f:
            json.dump(episode_data, f, cls=EnhancedJSONEncoder)
    except TypeError as e:
        # If still having issues, try additional serialization steps
        print(f"Error serializing JSON: {e}")
        print("Attempting to fix serialization issues...")

        # Convert all objects in episode_data to ensure they're serializable
        safe_episode_data = []
        for timestep_data in episode_data:
            # Process each timestep separately
            try:
                safe_timestep = ensure_serializable(timestep_data)
                safe_episode_data.append(safe_timestep)
            except Exception as e2:
                print(f"Error processing timestep data: {e2}")
                # Create a simplified version of the timestep data
                safe_timestep = {
                    "timestep": timestep_data.get("timestep", 0),
                    "info": {
                        "error": f"Failed to serialize full data: {str(e2)}",
                        "success": timestep_data.get("info", {}).get("success", False)
                    }
                }
                safe_episode_data.append(safe_timestep)

        # Try to save the sanitized data
        try:
            with open(episode_file_path, 'w') as f:
                json.dump(safe_episode_data, f)
        except Exception as e3:
            print(f"Critical error saving episode data: {e3}")
            # Last resort: save what we can
            with open(episode_file_path, 'w') as f:
                f.write('{"error": "Failed to serialize episode data"}')

    # Free memory by clearing episode data
    del episode_data

    # Update success count if this episode was successful
    if success:
        success_count += 1

    # Add minimal info to summary
    summary_data.append({
        "episode_id": episode_id,
        "success": success,
        "steps_taken": elapsed_steps
    })

    # Log progress
    if episode_id % 10 == 0 or episode_id == 49:
        log_progress(episode_id, 50, success_count)

    # Explicitly run garbage collection between episodes
    gc.collect()

# Save summary data in the same folder (smaller file, can use indentation)
summary_file_path = os.path.join(output_folder, 'episodes_summary.json')
with open(summary_file_path, 'w') as f:
    json.dump(summary_data, f, indent=2, cls=EnhancedJSONEncoder)

# Final stats
total_elapsed = (datetime.now() - start_time).total_seconds() / 60
print(f"Completed 50 episodes in {total_elapsed:.1f} minutes")
print(f"Data saved to {output_folder}/ folder")
print(f"Total successful episodes: {success_count}/50 ({success_count/50*100:.1f}%)")
print(f"Average steps taken: {np.mean([ep['steps_taken'] for ep in summary_data]):.2f}")

# Clear any remaining references to large data structures
del summary_data
gc.collect()
