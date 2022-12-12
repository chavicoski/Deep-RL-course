#!/bin/bash
# Use the run id of the model to push. By default it should be a date (%Y-%m-%d_%H-%M-%S)
export RUN_ID="2022-12-12_22-42-47"
# Use the repo id for your {user/organization}/{repo-name}
export REPO_ID="chavicoski/ppo-Huggy-test"
# A message to show with the model description
export COMMIT_MESSAGE="Test push with custom script"

mlagents-push-to-hf \
    --run-id=$RUN_ID \
    --local-dir="./runs/train/${RUN_ID}" \
    --repo-id=$REPO_ID \
    --commit-message=$COMMIT_MESSAGE
