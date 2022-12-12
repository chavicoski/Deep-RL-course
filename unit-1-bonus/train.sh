#!/bin/bash
export RUN_ID=$(date '+%Y-%m-%d_%H-%M-%S')
export RESULTS_DIR=./src/runs/train

mlagents-learn ./src/Huggy_train_params.yaml \
    --env=ml-agents/trained-envs-executables/linux/Huggy/Huggy \
    --run-id=$RUN_ID \
    --results-dir $RESULTS_DIR \
    --no-graphics
