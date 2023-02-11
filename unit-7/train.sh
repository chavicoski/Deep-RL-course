#!/bin/bash
mlagents-learn ./src/SoccerTwos_params.yaml \
    --env=ml-agents/training-envs-executables/linux/SoccerTwos/SoccerTwos.x86_64 \
    --run-id=$(date '+%Y-%m-%d_%H-%M-%S') \
    --results-dir=./src/runs/train \
    --no-graphics
