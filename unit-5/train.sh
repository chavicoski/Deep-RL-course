#!/bin/bash
export ENV="Pyramids" # Options: "SnowballTarget" and "Pyramids"
mlagents-learn ./src/${ENV}_params.yaml \
    --env=ml-agents/training-envs-executables/linux/$ENV/$ENV \
    --run-id=$(date '+%Y-%m-%d_%H-%M-%S') \
    --results-dir=./src/runs/train \
    --no-graphics
