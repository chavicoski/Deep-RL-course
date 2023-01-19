"""
Script to upload a model checkpoint (generated with `train.py`) to the Hugging Face hub.
You should configure the file `config/model_to_hub.yaml` to push your model
"""
import gym
import hydra
import panda_gym  # Add envs to `gym`
import pybullet_envs  # Add envs to `gym`
from huggingface_sb3 import package_to_hub
from hydra.utils import instantiate
from omegaconf import DictConfig
from pyvirtualdisplay import Display
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


@hydra.main(version_base=None, config_path="config", config_name="model_to_hub")
def main(cfg: DictConfig):
    # Load model from checkpoint zip file
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(cfg.env_id))])
    model: BaseAlgorithm = instantiate(cfg.load_model, env=eval_env)

    # We need to setup a display to push the agent with a video demo
    virtual_display = Display(visible=0, size=(1400, 900))
    virtual_display.start()

    # Push the model to the hub
    package_to_hub(
        model=model,
        model_name=cfg.model_name,
        model_architecture=cfg.model_architecture,
        env_id=cfg.env_id,
        eval_env=eval_env,
        repo_id=cfg.repo_id,
        commit_message=cfg.commit_message,
    )


if __name__ == "__main__":
    main()
