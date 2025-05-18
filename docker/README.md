# Docker 

# Overview
In the Breast Cancer Classification project we have two docker files.

- `train.dockerfile` - The train image is run when we want to train the model.
- `predict.dockerfile` - The predict image is run when we want to generate predictions.

# Docker Installation Instructions

## OS Caveats

# Running a Docker Image


# Detailed Information


## `train.dockerfile` 

The train dockerfile 


## `predict.dockerfile` 

## `train.dockerfile` size vs `predict.dockerfile` size

Traditionally we would expect to have our train and predict dockerfiles to be smaller than one another. However, breast-cancer-classification was built to act as a python module. Due to this we cannot easily separate out particular scripts or package requirements. As such both train and predict have the same size at ~1.0GB.
