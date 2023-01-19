![Env](etc/environments.gif)

# Unit 6: Advantage Actor-Critic

You can find the unit theory [here](https://huggingface.co/deep-rl-course/unit6/introduction?fw=pt).

## HOW TO RUN

First check that you have the requirements listed [here](../README.md).

**IMPORTANT**: Before runing the following commands ensure that you are inside the unit folder.

### 1 - Build the Docker image

You only need to run this command once

    docker-compose build

### 2 - Run training

If you want to modify some hyperparameters you can do it in the configuration files [a2c.yaml](config/model/a2c.yaml) and [train.yaml](config/train.yaml).

By default it will train an agent in the *AntBullet-v0* environment. You can select the *PandaReachDense-v2* environment by selecting the "*panda_reach_dense_v2*" env in the "*env*" field in [train.yaml](config/train.yaml). You should also change the "*experiment_name*" field to fit your experiment configuration.

    docker-compose run train
   
Each training run will create a folder in "*runs/train/\<DATETIME\>/*". You will find there the experiment outputs (like the trained model).
    
### 3 - Push the trained model to the Hub

Before pushing the model to the Hub **YOU MUST** edit the file [model_to_hub.yaml](config/model_to_hub.yaml). You should at least change the model path (to point to the model in "*runs/train/\<DATETIME\>/\<MODEL_FILENAME\>.zip*") and the *repo_id* to push the model to you personal repository. If you selected a different environment for training you should also change the "*env_id*" field.

    docker-compose run model_to_hub
