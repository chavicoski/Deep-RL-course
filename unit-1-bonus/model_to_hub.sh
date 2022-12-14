#!/bin/bash
# Use the run id of the model to push. By default it should be a date (%Y-%m-%d_%H-%M-%S)
export RUN_ID="2022-12-13_16-30-16"
# Use the repo id for your {user/organization}/{repo-name}
export REPO_ID="chavicoski/ppo-Huggy"
# A message to show with the model description
export COMMIT_MESSAGE="New version with 0.0001 lr"

mlagents-push-to-hf \
    --run-id="$RUN_ID" \
    --local-dir="./runs/train/$RUN_ID" \
    --repo-id="$REPO_ID" \
    --commit-message="$COMMIT_MESSAGE"
