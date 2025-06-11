import os
import subprocess
import simpler_env
import tensorflow as tf
import logging
import json
import numpy as np
import sapien.core as sapien
import sys
import copy
from collections import deque
from tqdm import tqdm
import mediapy

# Add open_x_embodiment to path, assuming it's in /opt/open_x_embodiment
sys.path.append("/opt/open_x_embodiment/models")
import rt1

# JAX/Flax imports
import jax
import jax.numpy as jnp
from flax.training import checkpoints
import tensorflow_hub as hub


# Suppress TensorFlow warnings and configure GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"GPU memory growth configuration error: {e}")

def get_rt_1_x_jax_checkpoint(ckpt_dir="./checkpoints"):
    """Download the RT-1-X JAX checkpoint."""
    ckpt_path = os.path.join(ckpt_dir, "rt_1_x_jax")
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_dir, exist_ok=True)
        gs_path = "gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_jax"
        try:
            print(f"Downloading RT-1-X JAX checkpoint from {gs_path}...")
            subprocess.run(["gsutil", "-m", "cp", "-r", gs_path, ckpt_dir], check=True)
            print("✅ Download complete.")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading from {gs_path}: {e}")
            raise
    return ckpt_path

# --- Policy Class and Helper Functions from Notebook ---

class RT1Policy:
    def __init__(self, checkpoint_path=None, model=None, variables=None, seqlen=15, rng=None):
        self.model = model
        self._checkpoint_path = checkpoint_path
        self.seqlen = seqlen
        self.rng = rng if rng is not None else jax.random.PRNGKey(0)
        self._run_action_inference_jit = jax.jit(self._run_action_inference)
        if variables:
            self.variables = variables
        else:
            state_dict = checkpoints.restore_checkpoint(checkpoint_path, None)
            self.variables = {
                'params': state_dict['params'],
                'batch_stats': state_dict['batch_stats'],
            }

    def _extract_action_info(self, output_logits):
        time_step_tokens = self.model.num_image_tokens + self.model.num_action_tokens
        output_logits = jnp.reshape(output_logits, (1, self.seqlen, time_step_tokens, -1))
        action_logits = output_logits[:, -1, ...]
        action_logits = action_logits[:, self.model.num_image_tokens - 1 : -1]

        action_logp = jax.nn.softmax(action_logits, axis=-1)
        token_argmax = jnp.argmax(action_logp, axis=-1)
        token_entropy = -jnp.sum(action_logp * jnp.log(action_logp + 1e-8), axis=-1)

        return {
            "token_probs": action_logp,
            "token_argmax": token_argmax,
            "token_entropy": token_entropy
        }

    def _run_action_inference(self, observation, rng):
        act_tokens = jnp.zeros((1, 6, 11))
        batch_obs = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), observation)
        _, random_rng = jax.random.split(rng)
        output_logits_tuple = self.model.apply(
            self.variables,
            batch_obs,
            act=None,
            act_tokens=act_tokens,
            train=False,
            rngs={'random': random_rng},
        )
        output_logits = output_logits_tuple[0]
        return self._extract_action_info(output_logits)

    def action(self, observation, use_model, task_description=None):
        observation = copy.deepcopy(observation)
        if task_description:
            embedding = use_model([task_description])[0].numpy()
            embedding = np.tile(embedding[None, :], (15, 1))
            observation['natural_language_embedding'] = embedding
        image = observation['image']
        image = tf.image.resize(image, (300, 300)).numpy() / 255.0
        observation['image'] = image
        results = []
        self.rng, rng = jax.random.split(self.rng)
        result = self._run_action_inference_jit(observation, rng)
        results.append({
                    "token_probs": jax.device_get(result["token_probs"])[0],
                    "token_argmax": jax.device_get(result["token_argmax"])[0],
                    "token_entropy": jax.device_get(result["token_entropy"])[0],
                })
        return results

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def collect_trajectory(
    env, policy, use_model, task_description, steps=100,
):
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    obs, reset_info = env.reset(seed=np.random.randint(0, 1e6))
    print("Reset info", reset_info)
    frames, trajectory = [], []
    
    image_buffer = deque(maxlen=15)

    for t in tqdm(range(steps), desc="Executing episode"):
        image = get_image_from_maniskill2_obs_dict(env, obs)
        image_buffer.append(image)

        if len(image_buffer) < 15:
            stacked_images = np.stack([image_buffer[0]] * (15 - len(image_buffer)) + list(image_buffer))
        else:
            stacked_images = np.stack(image_buffer)

        obs_dict = {"image": stacked_images.astype(np.uint8)}
        results = policy.action(obs_dict, use_model, task_description)
        
        timestep_log = {"timestep": t}
        timestep_log["token_argmax"] = [a["token_argmax"].tolist() for a in results]
        timestep_log["token_entropy"] = [a["token_entropy"].tolist() for a in results]

        token_argmax = [a["token_argmax"] for a in results]
        detokenized_actions = [
            {k: v[0] for k, v in rt1.detokenize_action(
                t[None, :],
                policy.model.vocab_size,
                policy.model.world_vector_range
            ).items()} for t in token_argmax
        ]
        
        mean_action = {
            k: np.mean([a[k] for a in detokenized_actions], axis=0)
            for k in detokenized_actions[0]
        }
        
        act_array = np.concatenate([
            mean_action['world_vector'],
            mean_action['rotation_delta'],
            mean_action['gripper_closedness_action']
        ])
        
        obs, reward, done, truncated, info = env.step(act_array)
        
        frames.append(image)
        timestep_log["mean_action"] = {k: v.tolist() for k, v in mean_action.items()}
        timestep_log["all_actions"] = [{k: v.tolist() for k, v in a.items()} for a in detokenized_actions]
        timestep_log["info"] = info
        trajectory.append(timestep_log)

        if done or truncated:
            print(f"Episode terminated at step {t} with success: {info.get('success', False)}")
            break

    return frames, trajectory, info.get('success', False)


def main():
    # --- Configuration ---
    task_name = "google_robot_pick_horizontal_coke_can"
    task_description = "Pick up the coke can"
    num_episodes = 50
    max_steps_per_episode = 400
    output_dir = f"rt900k_{task_name}_results"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "json"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "video"), exist_ok=True)

    # --- Setup ---
    print("Setting up environment...")
    env = simpler_env.make(task_name)
    sapien.render_config.rt_use_denoiser = False

    print("Downloading checkpoint...")
    checkpoint_path = get_rt_1_x_jax_checkpoint()

    print("Loading Universal Sentence Encoder...")
    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    print("Initializing RT-1-X (JAX) model and policy...")
    rt1x_model = rt1.RT1(
        num_image_tokens=81,
        num_action_tokens=11,
        layer_size=256,
        vocab_size=512,
        use_token_learner=True,
        world_vector_range=(-2.0, 2.0),
    )
    policy = RT1Policy(
        checkpoint_path=checkpoint_path,
        model=rt1x_model,
        seqlen=15,
    )

    # --- Main Loop ---
    success_count = 0
    for i in range(num_episodes):
        print(f"--- Starting Episode {i+1}/{num_episodes} ---")
        
        frames, trajectory, success = collect_trajectory(
            env,
            policy,
            use_model,
            task_description,
            steps=max_steps_per_episode,
        )
        
        if success:
            success_count += 1
        
        # Save results
        filename_prefix = f"{str(success)}_{i}"
        json_path = os.path.join(output_dir, "json", f"{filename_prefix}.json")
        video_path = os.path.join(output_dir, "video", f"{filename_prefix}.mp4")

        with open(json_path, "w") as f:
            json.dump(convert_numpy(trajectory), f, indent=2)
        
        mediapy.write_video(video_path, frames, fps=10)
        
        print(f"Episode {i+1} finished. Success: {success}. Results saved to {output_dir}")
        print(f"Current success rate: {success_count}/{i+1} ({(success_count/(i+1))*100:.2f}%)")

    env.close()
    print("\n--- All Episodes Complete ---")
    print(f"Final success rate: {success_count}/{num_episodes} ({(success_count/num_episodes)*100:.2f}%)")


if __name__ == '__main__':
    main()
