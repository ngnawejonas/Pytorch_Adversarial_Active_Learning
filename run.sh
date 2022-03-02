#!/bin/bash
git clone https://github.com/ngnawejonas/Pytorch_Adversarial_Active_Learning
python Pytorch_Adversarial_Active_Learning/active_framework.py --id_experiment=$1 --attack=$2 --n_pool=$3