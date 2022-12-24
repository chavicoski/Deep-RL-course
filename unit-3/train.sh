#!/bin/bash
export ALGO="dqn"
export ENV_NAME="SpaceInvadersNoFrameskip-v4"
export LOG_DIR="save_dir/logs"
export TRAIN_HPARAMS="model_hparams.yml"
export EVAL_TIMESTEPS=5000

# Train the agent
python rl-baselines3-zoo/train.py \
    --algo $ALGO \
    --env $ENV_NAME \
    -f $LOG_DIR \
    -conf $TRAIN_HPARAMS

# Evaluate the agent
echo "Going to evaluate the model"
python rl-baselines3-zoo/enjoy.py \
    --algo $ALGO \
    --env $ENV_NAME \
    -f $LOG_DIR \
    --n-timesteps $EVAL_TIMESTEPS \
    --no-render
