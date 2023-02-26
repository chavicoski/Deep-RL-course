import functools

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.enjoy import enjoy
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
from sf_examples.vizdoom.doom.doom_params import (add_doom_env_args,
                                                  doom_override_defaults)
from sf_examples.vizdoom.doom.doom_utils import (DOOM_ENVS,
                                                 make_doom_env_from_spec)


def register_vizdoom_envs():
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)


def register_vizdoom_models():
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)


def register_vizdoom_components():
    register_vizdoom_envs()
    register_vizdoom_models()


# Create a config from command line args
def parse_vizdoom_cfg(argv=None, evaluation=False):
    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    # Parameters specific to Doom envs
    add_doom_env_args(parser)
    # Override Doom default values for algo parameters
    doom_override_defaults(parser)
    # Second parsing pass yields the final configuration
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():
    register_vizdoom_components()

    # Select the environment to train
    env = "doom_health_gathering_supreme"
    # Performance config
    n_workers = 14
    n_envs_per_worker = 8
    # Train hyperparameters
    n_steps = 100000000
    # Full list of parameters with description:
    # https://www.samplefactory.dev/02-configuration/cfg-params/

    # Create the experiment config
    cfg = parse_vizdoom_cfg(
        argv=[
            f"--env={env}",
            f"--num_workers={n_workers}",
            f"--num_envs_per_worker={n_envs_per_worker}",
            f"--train_for_env_steps={n_steps}",
            "--train_dir=./runs",
        ]
    )
    # Train
    run_rl(cfg)

    # Insert the HuggingFace username to upload the model to the hub
    hf_username = "chavicoski"

    cfg = parse_vizdoom_cfg(
        argv=[
            f"--env={env}",
            "--num_workers=1",
            "--save_video",
            "--no_render",
            "--max_num_episodes=10",
            "--max_num_frames=100000",
            "--push_to_hub",
            f"--hf_repository={hf_username}/vizdoom_health_gathering_supreme",
        ],
        evaluation=True,
    )
    enjoy(cfg)


if __name__ == "__main__":
    main()
