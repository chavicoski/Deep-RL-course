"""
Main script for training a Reinforce agent implemented from scratch using Pytorch.
The script can be configured to change the environment, the hyperparameters...
The configuration is handled using the `hydra` library. You can modify the configuration
using the YAML files in the `config` folder
"""
import json

import hydra
import torch
from gym import Env
from hydra.utils import instantiate
from lib.reinforce import Policy, evaluate_agent, record_video, reinforce
from lib.utils import get_device
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    # Select the available computing device
    device = get_device(cfg.device)

    # Prepare the training environment
    env: Env = instantiate(cfg.train_env)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create the policy to train
    policy: Policy = instantiate(
        cfg.policy, state_size=state_size, action_size=action_size
    ).to(
        device  # Move the policy to the selected device
    )

    # Parameters optimizer
    opt = instantiate(cfg.optimizer, policy.parameters())

    # Train phase
    print(f"Starting training for {cfg.train_hparams.n_training_episodes} episodes")
    reinforce(env, policy, opt, device=device, **cfg.train_hparams)

    # Prepare the evaluation environment
    eval_env = instantiate(cfg.eval_env)
    print("Evaluating the model...")
    mean_reward, std_reward = evaluate_agent(
        eval_env, policy, device=device, **cfg.eval_hparams
    )
    print(f"Evaluation results: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Store the Policy model to a file
    torch.save(policy, cfg.model_path)
    # Create a demo video of the trained agent
    record_video(eval_env, policy, cfg.video_path, cfg.video_fps, device)
    # Save hyperparameters
    with open(cfg.hparams_path, "w") as outfile:
        json.dump(OmegaConf.to_container(cfg), outfile)


if __name__ == "__main__":
    main()
