import hydra
from gym import Env
from hydra.utils import instantiate
from omegaconf import DictConfig
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    env: Env = instantiate(cfg.train_env)
    model: BaseAlgorithm = instantiate(cfg.model, env=env)

    # Train
    model.learn(total_timesteps=cfg.total_timesteps)
    model.save(cfg.experiment_name)

    # Evaluate
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
