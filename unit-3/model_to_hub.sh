#!/bin/bash
# Model algorithm used
export ALGO="dqn"
# Environment used to train the agent
export ENV_NAME="SpaceInvadersNoFrameskip-v4"
# Log dir with the trained model (relative to the "rl-baselines3-zoo" dir)
export LOG_DIR="../save_dir/logs"
# User/organization to store the model in the Hub
export ORGANIZATION="chavicoski"
# Repository name in the Hub
export REPO="dqn-SpaceInvadersNoFrameskip-v4"

cd rl-baselines3-zoo
python -m rl_zoo3.push_to_hub \
    --algo $ALGO \
    --env $ENV_NAME \
    -f $LOG_DIR \
    -orga $ORGANIZATION \
    --repo-name $REPO
