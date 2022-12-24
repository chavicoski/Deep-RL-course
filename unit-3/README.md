![Atari Envs](etc/atari-envs.gif)

# Unit 3: Deep Q-Learning with Atari Games

You can find the unit theory [here](https://huggingface.co/deep-rl-course/unit3/introduction?fw=pt).

## HOW TO RUN

First check that you have the requirements listed [here](../README.md).

**IMPORTANT**: Before runing the following commands ensure that you are inside the unit folder.

### 1 - Build the Docker image

You only need to run this command once

    docker-compose build

### 2 - Run training

If you want to modify some hyperparameters you can do it in the configuration file [model_hparams.yml](model_hparams.yml).

    docker-compose run train
   
Each training run will create a folder in "*logs/dqn/\<RUN_ID\>/*". You will find there the experiment outputs (like the trained model, hyperparameters used...).
    
### 3 - Push the trained model to the Hub

Before pushing the model to the Hub **YOU MUST** edit the file [model_to_hub.sh](model_to_hub.sh). You should at least change the *ORGANIZATION* to push the model to you personal repository.

    docker-compose run model_to_hub
