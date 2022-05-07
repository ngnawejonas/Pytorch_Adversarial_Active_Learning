# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:03:13 2017

@author: mducoffe
@edit:ngnawejonas
"""
import os
import sys
import time
import csv
import argparse
from contextlib import closing
import yaml
import gc

import numpy as np

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch.distributions import Categorical

from build_model import build_model_func
from build_data import build_data_func, getSize, MyDataSet
from train_lib import train_model, simple_test, get_optimzer, robust_test
from adversarial_active_criterion import Adversarial_DeepFool

#pylint: disable=invalid-name
#pylint: disable=too-many-arguments

SHOW_LOG = True


def log(msg, *args):
    """Logging function"""
    if SHOW_LOG:
        print(msg, *args)

# %%
# %%


def active_training(labelled_data, model=None, attack=None):
    """active training."""
    # split into train and validation

    n_train = (int)(len(labelled_data) * 0.8)

    best_model = None
    best_loss = np.inf
    for i in range(REPEAT):
        log(f'training No {i+1}')
        # shuffle DATASET_NAME and split train and val
        train_data, val_data = random_split(
            labelled_data, [n_train, len(labelled_data) - n_train])
        if model is None:
            model = build_model_func(NETWORK_ARCH, IMG_SIZE)

        model = model.to(DEVICE)
        optz = get_optimzer(model, NETWORK_ARCH, DATASET_NAME)

        loss = train_model(train_data=train_data,
                           validation_data=val_data,
                           model=model, epochs=EPOCHS, batch_size=BATCH_SIZE,
                           optz=optz, device=DEVICE, attack=attack)

        if loss < best_loss:
            best_loss = loss
            best_model = model

        # Clear GPU memory in preparation for next model training
        del model
        del loss
        gc.collect()
        torch.cuda.empty_cache()
    return best_model

# %%


def evaluate(
        model,
        test_data,
        percentage):
    """Evaluate model."""
    timer = time.time()
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
    acc = simple_test(test_dataloader, model.to(DEVICE), device=DEVICE)
    acc_r = 0
    #robust_test(
        # test_dataloader,
        # model.to(DEVICE),
        # attack=ATTACK,
        # device=DEVICE)
    timer = time.time() - timer
    log("eval time: {:.2f}".format(timer))
    with closing(open(os.path.join(REPO, FILENAME), 'a')) as csvfile:
        # TO DO
        writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([str(ID_EXP), str(percentage), str(acc), str(acc_r)])


def load():
    """Loading dataset and model."""

    model = build_model_func(NETWORK_ARCH, IMG_SIZE)

    labelled_data, unlabelled_data, test_data, full_train = build_data_func(
        DATASET_NAME, INITIAL_SAMPLE, SEED)

    return model, labelled_data, unlabelled_data, test_data, full_train


def active_selection(model, unlabelled_data, nb_data, strategy):
    """Active selection."""
    if strategy == 'uncertainty':
        query, unlabelled_data = uncertainty_selection(
            model, unlabelled_data, nb_data)
    elif strategy in ['random', 'adv_train']:
        query, unlabelled_data = random_selection(
            model, unlabelled_data, nb_data)
    elif strategy == 'aaq':
        query, unlabelled_data = adversarial_selection(
            model, unlabelled_data, nb_data, False)
    elif strategy == 'saaq':
        query, unlabelled_data = adversarial_selection(
            model, unlabelled_data, nb_data, True)
    else:
        return NotImplementedError(f'Unknown active criterion {strategy}')

    return query, unlabelled_data


def random_selection(model, unlabelled_data, nb_data):
    """Random selection strategy."""
    # select a random subset
    subset_index = np.random.choice(
        unlabelled_data.indices,
        size=nb_data,
        replace=False)
    subset = Subset(unlabelled_data.dataset, subset_index)

    # compute distances to adv attacks on subset
    # deepfool = Adversarial_DeepFool(model=model, device=DEVICE)
    # _idx, _adv = deepfool.generate(subset, ATTACK, diversity=False)

    # remove selected indexes from unlabelled_data
    for idx in subset.indices:
        unlabelled_data.indices.remove(idx)

    dtl = DataLoader(subset, batch_size=len(subset))
    for X, y in dtl:
        new_data = MyDataSet(data=X, targets=y)

    return new_data, unlabelled_data


def uncertainty_selection(model, unlabelled_data, nb_data):
    """Uncertainty strategy."""

    n = len(unlabelled_data.indices)# min(300, len(unlabelled_data.indices))
    subset_index = np.random.choice(
        unlabelled_data.indices, size=n, replace=False) # not really necessary if == len(unlabelled_data)
    # log(n, subset_index)
    subset = Subset(unlabelled_data.dataset, subset_index)

    entropies = []
    dataloader = DataLoader(subset)
    model.eval()
    for image, _ in dataloader:
        image = image.to(DEVICE)
        probabilities = torch.softmax(model(image), dim=1)
        entropy = Categorical(probs=probabilities).entropy()
        entropies.append(entropy.detach())

    entropies = torch.Tensor(entropies)
    index = entropies.argsort(descending=True)

    chosen_indices = [unlabelled_data.indices[i] for i in index[:nb_data]]
    # remove selected indexes from unlabelled_data
    for idx in chosen_indices:
        unlabelled_data.indices.remove(idx)

    subset = Subset(unlabelled_data.dataset, chosen_indices)
    dtl = DataLoader(subset, batch_size=len(subset))
    for X, y in dtl:
        new_data = MyDataSet(data=X, targets=y)

    return new_data, unlabelled_data


def adversarial_selection(model, unlabelled_data, nb_data, add_adv=False):
    """Adversarial strategy."""

    # select a subset of size 10*nb_data
    n = min(300, len(unlabelled_data.indices))
    subset_index = np.random.choice(
        unlabelled_data.indices, size=n, replace=False)
    # log(n, subset_index)
    subset = Subset(unlabelled_data.dataset, subset_index)

    # compute distances to adv attacks on subset
    deepfool = Adversarial_DeepFool(model=model, device=DEVICE)
    chosen_indices, attacked_images = deepfool.generate(
        subset, ATTACK, diversity=DIVERSITY)

    # get selected images
    subset.indices = chosen_indices[:nb_data]

    # remove selected indexes from unlabelled_data
    for idx in subset.indices:
        unlabelled_data.indices.remove(idx)

    dtl = DataLoader(subset, batch_size=len(subset))
    for X, y in dtl:
        new_data = MyDataSet(data=X, targets=y)

    if add_adv:
        # log(type(unlabelled_data.dataset.targets), type(chosen_indices))
        if isinstance(unlabelled_data.dataset.targets, list):
            labels = torch.tensor(
                unlabelled_data.dataset.targets)[
                subset.indices]
        else:
            labels = unlabelled_data.dataset.targets[subset.indices]
        new_data.add_data(attacked_images[:nb_data], labels)

    return new_data, unlabelled_data


# %%
def active_learning():
    """Active learning"""

    # Phase 1: Initial step
    log('Phase 1: Loading & spliting dataset')
    model, labelled_data, unlabelled_data, test_data, full_train = load()

    active_train_attack = None
    if ACTIVE_METHOD == 'adv_train':
        active_train_attack = ATTACK

    percentage_data = INITIAL_SAMPLE  # len(labelled_data)
    log('START')
    timer0 = time.time()
    while percentage_data < POOL_SIZE:
        log('percentage_data = ', percentage_data)

        # Phase 2: Active Training
        log('Phase 2: Active Training')
        model = active_training(
            labelled_data,
            model,
            attack=active_train_attack)

        # phase 3: 
        timer = time.time()
        log(f"Phase 3: Active selection: {ACTIVE_METHOD}")
        query, unlabelled_data = active_selection(
            model, unlabelled_data, QUERY_SIZE, ACTIVE_METHOD)
        # add query to the labelled set
        labelled_data.cat(query)
        # update percentage_data
        percentage_data = len(labelled_data)
        timer = time.time() - timer
        log("{}: active selection time {:.2f} seconds.".format(
            percentage_data, timer))

        # Phase 4:
        log("Phase 4: Evaluate and report test acc of model")
        evaluate(model, test_data, percentage_data)

    log('percentage_data = ', percentage_data)
    model = active_training(
        labelled_data,
        model=None,
        attack=active_train_attack)
    log("Phase 4: Evaluate and report test acc of model")
    evaluate(
        model,
        test_data,
        percentage_data)
    timer = time.time() - timer0
    log("END: {:.2f}".format(timer))

    # timer = time.time()
    # log('Training on random sample of same size')
    # # subset_index = np.random.choice(full_train.indices, size=percentage_data, replace=False)
    # # random_subset = Subset(full_train.dataset, subset_index)
    # random_subset, _ = random_split(
    #     full_train, [
    #         percentage_data, len(full_train) - percentage_data])
    # model = active_training(
    #     random_subset,
    #     model=None,
    #     attack=active_train_attack)
    # log("1/Evaluate and report test acc of random sample of same size")
    # evaluate(model, test_data, percentage_data)

    # log('adversarial training on random sample of same size')
    # model = active_training(
    #     random_subset,
    #     model=None,
    #     attack=active_train_attack)
    # log("2/Evaluate and report test acc of random sample of same size (adv train)")
    # evaluate(model, test_data, percentage_data)

    # log('training on full data')
    # model = active_training(full_train, model=None, attack=active_train_attack)
    # log("3/Evaluate and report test acc of full DATASET_NAME model")
    # evaluate(model, test_data, len(full_train))

    # log('adversarial training on full DATASET_NAME')
    # model = active_training(full_train, model=None, attack=active_train_attack)
    # log("4/Evaluate and report test acc of full data model (adv train)")
    # evaluate(model, test_data, len(full_train))
    # timer = time.time() - timer
    # log("ADDITIONAL EVALS: {:.2f}".format(timer))


# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Active Learning')

    parser.add_argument(
        '--id_experiment',
        type=int,
        default=4,
        help='id number of experiment')
    parser.add_argument(
        '--repo',
        type=str,
        default='.',
        help='Repsitory for log')
    parser.add_argument(
        '--initial_sample',
        type=int,
        default=100,
        help='size of the initial training set')
    parser.add_argument(
        '--n_pool',
        type=int,
        default=200,
        help='Final train size')
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='mnist',
        help='dataset')
    parser.add_argument(
        '--net_arch',
        type=str,
        default='resnet18',
        help='name of the network architecture')
    parser.add_argument(
        '--active',
        type=str,
        default='saaq',
        help='active techniques')
    parser.add_argument(
        '--attack',
        type=str,
        default='fgsm',
        help='type of attack')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument(
        '--repeat',
        type=int,
        default=2,
        help='train repeats for active training')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='if True(>0) the initial 100 images will be the same everytime')
    parser.add_argument(
        '--diversity',
        type=int,
        default=0,
        help='if True(>0) diversity selection applied')

    args_ = parser.parse_args()

    if len(sys.argv)> 1:
        ID_EXP = args_.id_experiment
        REPO = args_.repo
        DATASET_NAME = args_.dataset_name
        NETWORK_ARCH = args_.net_arch
        ACTIVE_METHOD = args_.active
        INITIAL_SAMPLE = args_.initial_sample
        POOL_SIZE = args_.n_pool
        ATTACK = args_.attack
        BATCH_SIZE = args_.batch_size
        EPOCHS = args_.epochs
        REPEAT = args_.repeat
        SEED = args_.seed
        DIVERSITY = args_.diversity
        QUERY_SIZE = 100
    else:
        try:
            with open('../config.yaml', 'r') as config_file:
                config = yaml.load(config_file, Loader=yaml.SafeLoader)
                ID_EXP = config['id_experiment']
                REPO = config['repo']
                DATASET_NAME = config['dataset_name']
                NETWORK_ARCH = config['network_arch']
                ACTIVE_METHOD = config['active_method']
                INITIAL_SAMPLE = config['initial_sample_size']
                POOL_SIZE = config['final_pool_size']
                ATTACK = config['attack']
                BATCH_SIZE = config['batch_size']
                EPOCHS = config['epochs']
                REPEAT = config['repeat']
                SEED = config['seed']
                DIVERSITY = config['diversity']
                QUERY_SIZE = config['query_size']
            log('importing params from config file')
            log(config)
        except yaml.YAMLError as exc:
            print("Error in configuration file:", exc)

    IMG_SIZE = getSize(DATASET_NAME)
    FILENAME = 'acc_{}_{}_{}_{}_{}'.format(
        DATASET_NAME, NETWORK_ARCH, ACTIVE_METHOD, POOL_SIZE, ATTACK)

    # Get cpu or gpu DEVICE for training.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {DEVICE} device")

    timer_ = time.time()

    active_learning()

    timer_ = time.time() - timer_
    log('Total Time: {:.2f} seconds'.format(timer_))
    f = open("time.txt", 'a')
    f.write('Time: {:.2f} seconds'.format(timer_))
    f.close()
