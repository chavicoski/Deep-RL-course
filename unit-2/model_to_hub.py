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

import hydra
import pickle5 as pickle
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save
from lib.q_learning import QLearning
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="model_to_hub")
def main(cfg: DictConfig):
    # Load model from checkpoint zip file
    with open(cfg.model_path, "rb") as f:
        model: QLearning = pickle.load(f)
    # Load the eval hparams to evaluate the model again
    model_dir = os.path.dirname(cfg.model_path)
    with open(os.path.join(model_dir, "eval_hparams.json"), "r") as f:
        eval_hparams = json.load(f)

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
    shutil.copyfile(
        os.path.join(model_dir, "train_hparams.json"),
        repo_local_path / "train_hparams.json",
    )
    shutil.copyfile(
        os.path.join(model_dir, "eval_hparams.json"),
        repo_local_path / "eval_hparams.json",
    )

    # Evaluate the model and save the results
    print("Evaluating the model:")
    mean_reward, std_reward = model.evaluate(
        cfg.eval_episodes, eval_hparams["max_steps"], eval_hparams["seeds"]
    )

    evaluate_data = {
        "env_id": model.env.spec.id,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "n_eval_episodes": eval_hparams["n_episodes"],
        "eval_datetime": datetime.datetime.now().isoformat(),
    }
    # Store the evaluation results in a JSON file
    with open(repo_local_path / "results.json", "w") as f:
        json.dump(evaluate_data, f)

    # Create the model card for the Hub
    env_name = model.env.spec.id
    if "map_name" in model.env.spec.kwargs:
        env_name += "-" + model.env.spec.kwargs["map_name"]
        if model.env.spec.kwargs.get("is_slippery", "") == False:
            env_name += "-no_slippery"

    model_card = f"""
  # **Q-Learning** Agent playing1 **{cfg.env_id}**
  This is a trained model of a **Q-Learning** agent playing **{cfg.env_id}** .

  ## Usage

  ```python
  
  model = load_from_hub(repo_id="{cfg.repo_id}", filename="{model_filename}")

  # Don't forget to check if you need to add additional attributes (is_slippery=False etc)
  env = gym.make("{cfg.env_id}")
  ```
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
        dataset_pretty_name=env_name,
        dataset_id=env_name,
    )
    # Merge all the metadata
    model_tags = OmegaConf.to_container(cfg.tags) + [env_name]
    metadata = {"tags": model_tags, **eval_metadata}
    # Save the metadata
    metadata_save(readme_path, metadata)

    # Create the sample video
    video_path = repo_local_path / "replay.mp4"
    model.record_video(video_path, eval_hparams["max_steps"], fps=cfg.fps)

    # Push everything to the Hub
    api.upload_folder(
        repo_id=cfg.repo_id,
        folder_path=repo_local_path,
        path_in_repo=".",
    )
    print(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")


if __name__ == "__main__":
    main()
