#!/bin/bash
mlagents-learn ./src/Huggy_train_params.yaml \
    --env=ml-agents/trained-envs-executables/linux/Huggy/Huggy \
    --run-id=$(date '+%Y-%m-%d_%H-%M-%S') \
    --results-dir=./src/runs/train \
    --no-graphics
