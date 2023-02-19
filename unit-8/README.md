![Lunar Lander](etc/lunar_lander.gif)

# Unit 8: Proximal Policy Optimization (Part 1)

You can find the unit theory [here](https://huggingface.co/deep-rl-course/unit8/introduction?fw=pt).

## HOW TO RUN

First check that you have the requirements listed [here](../README.md).

**IMPORTANT**: Before runing the following commands ensure that you are inside the unit folder.

### 1 - Build the Docker image

You only need to run this command once

    docker-compose build

### 2 - Run training and pusth the model

In this unit the train script also pushes the model to the hub after training. Use the following command to run it:

    docker-compose run train

If you want to modify some hyperparameters you can do it in [docker-compose.yml](docker-compose.yaml), changing the flags passed to the *main.py* script. You **MUST** change the *repo-id* flag to use your hugging face account.
