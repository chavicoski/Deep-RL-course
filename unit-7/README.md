![SoccerTwos](etc/soccertwos.png)

# Unit 7: Introduction to Multi-Agents

You can find the unit theory [here](https://huggingface.co/deep-rl-course/unit7/introduction?fw=pt).

## HOW TO RUN

First check that you have the requirements listed [here](../README.md).

**IMPORTANT**: Before runing the following commands ensure that you are inside the unit folder.

### 1 - Build the Docker image

You only need to run this command once

    docker-compose build

### 2 - Run training

If you want to modify the hyperparameters you can do it in the configuration files [SoccerTwos_params.yaml](SoccerTwos_params.yaml).

    docker-compose run train
   
Each training run will create a folder in "*runs/train/\<DATETIME\>/*". You will find there the experiment outputs (parameters used, model checkpoints...).
    
### 3 - Push the trained model to the Hub

Before pushing the model to the Hub **YOU MUST** edit the file [model_to_hub.sh](model_to_hub.sh). You should at least change the *RUN_ID* (*\<DATETIME\>* of your training run) and the *REPO_ID* to push the model to you personal repository.

    docker-compose run model_to_hub
