"""
Script to upload a model checkpoint (generated with `train.py`) to the Hugging Face hub.
The steps are the following:
    - Evaluates the model
    - Generates the model card for the Hub
    - Generates the sample video
    - Pushes everything to the Hub

You should configure the file `config/model_to_hub.yaml` to push your model
"""
import datetime
import json
import os
import shutil
from pathlib import Path

import gym_pygame  # Import it to add more envs to Gym
import hydra
import torch
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save
from hydra.utils import instantiate
from lib.reinforce import Policy, evaluate_agent, record_video
from lib.utils import get_device
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="model_to_hub")
def main(cfg: DictConfig):
    # Prepare the Hugging Face API
    api = HfApi()

    # Create the repository
    repo_url = api.create_repo(repo_id=cfg.repo_id, exist_ok=True)

    # Download the files from the repo
    repo_local_path = Path(snapshot_download(repo_id=cfg.repo_id))

    # Add the model to the repo
    model_filename = os.path.basename(cfg.model_path)
    shutil.copyfile(cfg.model_path, repo_local_path / model_filename)
    # Add the hyperparameters to the repo
    hparams_filename = os.path.basename(cfg.hparams_path)
    shutil.copyfile(cfg.hparams_path, repo_local_path / hparams_filename)

    # Select the available computing device
    device = get_device(cfg.device)

    # Load model to run evaluation
    model: Policy = torch.load(cfg.model_path).to(device)

    # Evaluate the model and save the results
    env = instantiate(cfg.eval_env)
    print("Evaluating the model...")
    mean_reward, std_reward = evaluate_agent(
        env,
        model,
        cfg.eval_hparams.n_eval_episodes,
        cfg.eval_hparams.max_steps,
        device,
    )

    env_id = env.spec.id
    evaluate_data = {
        "env_id": env_id,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "n_eval_episodes": cfg.eval_hparams.n_eval_episodes,
        "eval_datetime": datetime.datetime.now().isoformat(),
    }
    # Store the evaluation results in a JSON file
    with open(repo_local_path / "results.json", "w") as f:
        json.dump(evaluate_data, f)

    # Create the model card for the Hub
    model_card = f"""
  # **Reinforce** Agent playing **{env_id}**
  This is a trained model of a **Reinforce** agent playing **{env_id}** .
  To learn to use this model and train yours check Unit 4 of the Deep Reinforcement Learning Course: https://huggingface.co/deep-rl-course/unit4/introduction
  """
    # Write the model card into the repo README
    readme_path = repo_local_path / "README.md"
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(model_card)

    # Get reward metric metadata from the model evaluation
    eval_metadata = metadata_eval_result(
        model_pretty_name=cfg.model_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_id,
        dataset_id=env_id,
    )
    # Merge all the metadata
    model_tags = OmegaConf.to_container(cfg.tags) + [env_id]
    metadata = {"tags": model_tags, **eval_metadata}
    # Save the metadata
    metadata_save(readme_path, metadata)

    # Create the sample video
    video_path = repo_local_path / "replay.mp4"
    record_video(env, model, video_path, cfg.fps, device)

    # Push everything to the Hub
    api.upload_folder(
        repo_id=cfg.repo_id,
        folder_path=repo_local_path,
        path_in_repo=".",
    )
    print(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")


if __name__ == "__main__":
    main()
