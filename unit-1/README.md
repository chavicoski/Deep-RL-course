![Lunar Lander](https://www.gymlibrary.dev/_images/lunar_lander.gif)

# Unit 1: Introduction to Deep Reinforcement Learning

You can find the unit theory [here](https://huggingface.co/deep-rl-course/unit1/introduction?fw=pt).

## HOW TO RUN

First check that you have the requirements listed [here](../README.md).

**IMPORTANT**: Before runing the following commands ensure that you are inside the unit folder.

### 1 - Build the Docker image

You only need to run this command once

    docker-compose build

### 2 - Run training

If you want to modify some hyperparameters you can do it in the configuration files [ppo.yaml](config/model/ppo.yaml) and [train.yaml](config/train.yaml).

    docker-compose run train
   
Each training run will create a folder in "*runs/train/\<DATETIME\>/*". You will find there the experiment outputs (like the trained model).
    
### 3 - Push the trained model to the Hub

Before pushing the model to the Hub **YOU MUST** edit the file [model_to_hub.yaml](config/model_to_hub.yaml). You should at least change the model path (to point to the model in "*runs/train/\<DATETIME\>/\<MODEL_FILENAME\>.zip*") and the *repo_id* to push the model to you personal repository.

    docker-compose run model_to_hub
