#!/bin/bash
git clone https://github.com/ngnawejonas/Pytorch_Adversarial_Active_Learning

for i in $(seq 0 1 $(($1-1))); do python Pytorch_Adversarial_Active_Learning/active_framework.py --id_experiment=$i --attack=$2 --n_pool=$3; done;