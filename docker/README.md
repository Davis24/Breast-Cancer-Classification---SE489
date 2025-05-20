# Docker 

## Overview
In the Breast Cancer Classification project we have two docker files.

- `train.dockerfile` - The train image is run when we want to train the model.
- `predict.dockerfile` - The predict image is run when we want to generate predictions.

## Docker Installation Instructions

While we highly recommend you follow the [instructions](https://docs.docker.com/get-started/get-docker/) provided by Docker itself. However we will detail some brief instructions below. 

- Install Docker GUI for your respective operating system. We recommend the GUI since it is easy to use.   The installation .exe can be found [here](https://docs.docker.com/get-started/get-docker/).
- Follow the .exe instructions. 
- Once installed restart your machine.
- `VSCode` - If you wish to include Docker information in your VSCode you can install the [VScode Docker extension](https://code.visualstudio.com/docs/containers/overview).


### OS Caveats

It should be noted that each OS has its own caveats, we will briefly detail them below.

- Windows - Requires WSL. 

## Building and Running a Docker Image

Before building or running the docker image make sure that the data in the `\data` folder exists. It should have already been run through the pre-processing steps which occurs during the `dataset.py` run. After this has occured you may proceed to using the train and predict docker images.

All commands should be exectued from the main directory: `\`

Note: You must build a docker image before you run it.

Building a docker image:
- Run `docker build --no-cache -f docker/<docker-file-name> . -t <tag-name>:latest`

Example: `docker build --no-cache -f train.dockerfile . -t train:latest`

Be patient this should take around 2-3 minutes to fully build.

Running:
- Run `docker run --name <name> <tag-name>:latest`  

Example: `docker run --name exp1 train:latest`

## Moving Files

In order to move files between docker images you will need to pull them from the docker image to your local. More information can be found [here](https://docs.docker.com/reference/cli/docker/container/cp/).

`docker cp {container_name}:{dir_path}/{file_name} {local_dir_path}/{local_file_name}`

## Additional Information

### `train.dockerfile` size vs `predict.dockerfile` size

Traditionally we would expect to have our train and predict dockerfiles to be smaller than one another. However, breast-cancer-classification was built to act as a python module. Due to this we cannot easily separate out particular scripts or package requirements. As such both train and predict have the same size at ~1.4GB.
