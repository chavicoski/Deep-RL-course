"""
Main script for training Deep Q-Learning models built from scratch using environments
from the `gym` library. The script can be fully configured to change the environment,
the hyperparameters...
The configuration is handled using the `hydra` library. You can modify the configuration
using the YAML files in the `config` folder
"""
import json

import hydra
import pickle5 as pickle
from gym import Env
from hydra.utils import instantiate
from lib.q_learning import QLearning
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    env: Env = instantiate(cfg.env)
    model = QLearning(env=env)

    # Train phase
    model.train(**cfg.train_hparams)

    # Save the trained model
    with open(f"{cfg.experiment_name}.pkl", "wb") as f:
        pickle.dump(model, f)
    # Save the training hyperparameters used
    with open(f"train_hparams.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg.train_hparams), f, indent=4)

    # Evaluate the trained model
    mean_reward, std_reward = model.evaluate(**cfg.eval_hparams)
    print(f"Evaluation results: {mean_reward=:.2f} +/- {std_reward:.2f}")
    # Save the evaluation hyperparameters used
    with open(f"eval_hparams.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg.eval_hparams), f, indent=4)


if __name__ == "__main__":
    main()
