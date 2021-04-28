# Pruning tutorial

This repository contains a small tutorial on pruning.
We are gonna train a simple feed forward network classifier on a toy dataset with two classes and features of dim=128.
Each epoch we will prune, using  a either structured layer-wise magnitude or unstructured globally magnitude criterion

- `main.py` defines the training and pruning schedule
- `model.py` defines the Network
- `PrunableModel.py` contains a class from which a model can inherit, such that it will create the needed variables for pruning. Additionally it has all the required functions.
- `./logs` will be a folder where tensorboard files will be generated. These function as logs and can be viewed by running `tensorboard --logdir ./logs`
 