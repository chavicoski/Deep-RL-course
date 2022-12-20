![Frozen Lake](https://www.gymlibrary.dev/_images/frozen_lake.gif)
![Taxi](https://www.gymlibrary.dev/_images/taxi.gif)

# Unit 2: Introduction to Q-Learning

You can find the unit theory [here](https://huggingface.co/deep-rl-course/unit2/introduction?fw=pt).

## HOW TO RUN

First check that you have the requirements listed [here](../README.md).

**IMPORTANT**: Before runing the following commands ensure that you are inside the unit folder.

### 1 - Build the Docker image

You only need to run this command once

    docker-compose build

### 2 - Run training

If you want to modify some hyperparameters you can do it in the configuration files [default.yaml](config/train_hparams/default.yaml). And to also configure the evaluation, you can modify the file [with_seed.yaml](config/eval_hparams/with_seed.yaml).

By default it will train an agent in the *Taxi-v3* environment. You can select the *Frozen Lake* environment by selecting the "*frozen_lake_v1*" env in the "*env*" field in [train.yaml](config/train.yaml). You should also change the "*experiment_name*" field to fit your experiment configuration.

    docker-compose run train
   
Each training run will create a folder in "*runs/train/\<DATETIME\>/*". You will find there the experiment outputs (like the trained model).
    
### 3 - Push the trained model to the Hub

Before pushing the model to the Hub **YOU MUST** edit the file [model_to_hub.yaml](config/model_to_hub.yaml). You should at least change the model path (to point to the model in "*runs/train/\<DATETIME\>/\<MODEL_FILENAME\>.zip*") and the *repo_id* to push the model to you personal repository. If you have changed the environment to train the model (to the *Frozen Lake*), you should also change the field that reference the env name.

    docker-compose run model_to_hub
