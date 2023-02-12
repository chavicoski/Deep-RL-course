#!/bin/bash
# Use the run id of the model to push. By default it should be a date (%Y-%m-%d_%H-%M-%S)
export RUN_ID="2023-02-11_11-39-53"
# Use the repo id for your {user/organization}/{repo-name}
export REPO_ID="chavicoski/poca-SoccerTwos"
# A message to show with the model description
export COMMIT_MESSAGE="First model. Trained for 15M steps. Achieved a training ELO around 1800"

mlagents-push-to-hf \
    --run-id="$RUN_ID" \
    --local-dir="./runs/train/$RUN_ID" \
    --repo-id="$REPO_ID" \
    --commit-message="$COMMIT_MESSAGE"
