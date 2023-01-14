#!/bin/bash
mlagents-learn ./src/SnowballTarget_params.yaml \
    --env=ml-agents/training-envs-executables/linux/SnowballTarget/SnowballTarget \
    --run-id=$(date '+%Y-%m-%d_%H-%M-%S') \
    --results-dir=./src/runs/train \
    --no-graphics
