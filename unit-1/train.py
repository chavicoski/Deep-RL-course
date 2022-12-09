"""
Main script for training reinforcement learning models (from `stable_baselines3`) using
environments from the `gym` library. The script can be fully configured to change the
model, the environment, the hyperparameters...
The configuration is handled using the `hydra` library. You can modify the configuration
using the YAML files in the `config` folder
"""
import hydra
from gym import Env
from hydra.utils import instantiate
from omegaconf import DictConfig
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    train_env: Env = instantiate(cfg.train_env)
    model: BaseAlgorithm = instantiate(cfg.model, env=train_env)

    # Train phase
    model.learn(total_timesteps=cfg.total_timesteps)
    model.save(cfg.experiment_name)

    # Evaluate the trained model
    eval_env: Env = instantiate(cfg.eval_env)
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=cfg.deterministic,
    )
    print(f"Evaluation results: {mean_reward=:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
