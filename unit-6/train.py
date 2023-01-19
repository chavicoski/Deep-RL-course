"""
Main script for training reinforcement learning models (from `stable_baselines3`) using
environments from the `PyBullet` and `Panda-Gym` libraries. The script can be fully
configured to change the model, the environment, the hyperparameters...
The configuration is handled using the `hydra` library. You can modify the configuration
using the YAML files in the `config` folder
"""
import gym
import hydra
import panda_gym  # Add envs to `gym`
import pybullet_envs  # Add envs to `gym`
from hydra.utils import instantiate
from omegaconf import DictConfig
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    # Prepare the training environment (with values normalization)
    train_env = make_vec_env(cfg.env.id, n_envs=cfg.env.n_envs)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model: BaseAlgorithm = instantiate(cfg.model, env=train_env)

    # Train phase
    model.learn(total_timesteps=cfg.total_timesteps)
    model.save(cfg.experiment_name)
    env_outfile = f"{cfg.experiment_name}_vec-norm-env.pkl"
    train_env.save(env_outfile)

    # Create the evaluation environment from the saved train_env
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(cfg.env.id))])
    eval_env = VecNormalize.load(env_outfile, eval_env)
    eval_env.training = False  # Disable training updates
    eval_env.norm_reward = False  # No need to norm the reward for evaluation
    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=cfg.deterministic,
    )
    print(f"Evaluation results: {mean_reward=:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
