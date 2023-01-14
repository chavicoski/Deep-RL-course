#!/bin/bash
# Use the run id of the model to push. By default it should be a date (%Y-%m-%d_%H-%M-%S)
export RUN_ID="2023-01-14_10-09-01"
# Use the repo id for your {user/organization}/{repo-name}
export REPO_ID="chavicoski/ppo-SnowballTarget"
# A message to show with the model description
export COMMIT_MESSAGE="SnowballTarget agent trained for 1000000 steps"

mlagents-push-to-hf \
    --run-id="$RUN_ID" \
    --local-dir="./runs/train/$RUN_ID" \
    --repo-id="$REPO_ID" \
    --commit-message="$COMMIT_MESSAGE"
