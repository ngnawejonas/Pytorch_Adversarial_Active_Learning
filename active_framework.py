# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:03:13 2017

@author: mducoffe
"""
import os
import time
import pickle
import csv

import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader, random_split, Subset, RandomSampler
from torch.distributions import Categorical

from contextlib import closing
from build_model import build_model_func
from build_data import build_data_func, getSize, MyDataSet
from train_lib import train_model, simple_test, get_optimzer, robust_test
from adversarial_active_criterion import Adversarial_DeepFool


SHOW_LOG = True

def log(msg, *args):
    if SHOW_LOG:
        print(msg, *args)

#%%
#%%
def active_training(labelled_data, network_name, img_size,
                    batch_size=64, epochs=100, repeat=2, device=None, attack=None):
    # split into train and validation
    
    N = len(labelled_data)
    n_train = (int) (N*0.8)

    batch_train = min(batch_size, n_train)
    steps_per_epoch = int(n_train/batch_train) + 1
    best_model = None
    best_loss = np.inf
    for i in range(repeat):
        log(f'training No {i+1}')
        # shuffle data and split train and val
        train_data, val_data = random_split(labelled_data, [n_train, N - n_train])
     
        model = build_model_func(network_name, img_size)
        model = model.to(device)
        optz = get_optimzer(model, network_name, data_name)

        loss = train_model(train_data=train_data,
                           validation_data=val_data,
                           model=model, epochs=epochs, batch_size=batch_size,
                           optz=optz, device=device, attack=attack)
                                   
        if loss < best_loss:
            best_loss = loss;
            best_model = model

    del model
    del loss

    return best_model

#%%
def evaluate(model, test_data, percentage, id_exp, repo, filename, device, batch_size):
    t = time.time()
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    acc = simple_test(test_dataloader, model.to(device), device=device)
    acc_r = robust_test(test_dataloader, model.to(device), attack=attack, device=device)
    t = time.time() - t
    log("eval time: {:.2f}".format(t))
    with closing(open(os.path.join(repo, filename), 'a')) as csvfile:
        # TO DO
        writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([str(id_exp), str(percentage), str(acc), str(acc_r)])
                             
                
def loading(num_sample, network_name, data_name, seed=False):
    img_size = getSize(data_name) # TO DO
    model=build_model_func(network_name, img_size)

    labelled_data, unlabelled_data, test_data, full_train = build_data_func(data_name, 
                                                                            num_sample=num_sample,
                                                                            seed=seed)
    
    return model, labelled_data, unlabelled_data, test_data, full_train


def active_selection(model, unlabelled_data, nb_data, active_method, attack, device, diversity=False):
    if active_method=='uncertainty':
        query, unlabelled_data = uncertainty_selection(model, unlabelled_data, nb_data)
    elif active_method in ['random', 'adv_train']:
        query, unlabelled_data = random_selection(model, unlabelled_data, nb_data, attack, device)
    elif active_method=='aaq':
        query, unlabelled_data = adversarial_selection(model, unlabelled_data, nb_data,
                                                         attack, False, device, diversity)
    elif active_method=='saaq':
        query, unlabelled_data = adversarial_selection(model, unlabelled_data, nb_data, 
                                                         attack, True, device, diversity)
    else:
        return NotImplementedError(f'Unknown active criterion {active_method}')

    return query, unlabelled_data
    
def random_selection(model, unlabelled_data, nb_data, attack, device):
    # select a random subset
    subset_index = np.random.choice(unlabelled_data.indices, size=nb_data*2, replace=False)
    subset = Subset(unlabelled_data.dataset, subset_index)

    # compute distances to adv attacks on subset
    active = Adversarial_DeepFool(model=model, device=device)
    _idx, _adv = active.generate(subset, attack, diversity=False)


    # remove selected indexes from unlabelled_data
    for idx in subset.indices:
        unlabelled_data.indices.remove(idx)    

    dtl = DataLoader(subset, batch_size=len(subset))
    for X,y in dtl:
        new_data = MyDataSet(data=X, targets=y)
    
    return new_data, unlabelled_data
           
def uncertainty_selection(model, unlabelled_data, nb_data):
    u_size = len(unlabelled_data.indices)
    n = min(300, u_size)
    subset_index = np.random.choice(unlabelled_data.indices, size=n, replace=False)
    # log(n, subset_index)
    subset = Subset(unlabelled_data.dataset, subset_index)

    entropies = []
    dataloader = DataLoader(subset)
    model.eval()
    for i, (image, _) in enumerate(dataloader):
        probabilities = torch.softmax(model(image), dim=1)        
        entropy = Categorical(probs = probabilities).entropy()
        entropies.append(entropy.detach())

    entropies = torch.Tensor(entropies)
    index = entropies.argsort(descending=True)

    chosen_indices = [unlabelled_data.indices[i] for i in index[:nb_data]]
    # remove selected indexes from unlabelled_data
    for idx in chosen_indices:
        unlabelled_data.indices.remove(idx)  

    subset = Subset(unlabelled_data.dataset, chosen_indices)
    dtl = DataLoader(subset, batch_size=len(subset))
    for X,y in dtl:
        new_data = MyDataSet(data=X, targets=y)

    return new_data, unlabelled_data

                 
def adversarial_selection(model, unlabelled_data, nb_data, attack='fgsm',
                         add_adv=False, device=None, diversity=False):

    # select a subset of size 10*nb_data
    u_size = len(unlabelled_data.indices)
    n = min(300, u_size)
    subset_index = np.random.choice(unlabelled_data.indices, size=n, replace=False)
    # log(n, subset_index)
    subset = Subset(unlabelled_data.dataset, subset_index)

    # compute distances to adv attacks on subset
    active = Adversarial_DeepFool(model=model, device=device)
    chosen_indices, attacked_images = active.generate(subset, attack, diversity=diversity)

    # get selected images
    subset.indices  = chosen_indices[:nb_data]

    # remove selected indexes from unlabelled_data
    for idx in subset.indices:
        unlabelled_data.indices.remove(idx)    

    dtl = DataLoader(subset, batch_size=len(subset))
    for X,y in dtl:
        new_data = MyDataSet(data=X, targets=y)

    if add_adv:
        # log(type(unlabelled_data.dataset.targets), type(chosen_indices))
        if type(unlabelled_data.dataset.targets) == list:
            labels = torch.tensor(unlabelled_data.dataset.targets)[subset.indices]
        else:
            labels = unlabelled_data.dataset.targets[subset.indices]
        new_data.add_data(attacked_images[:nb_data], labels)

    return new_data, unlabelled_data
               

#%%
def active_learning(num_sample, data_name, network_name, active_name, attack='pgd',
                    id_exp=0, nb_query=100, n_pool = 2000, repo='test', filename='test.csv',
                    device=None, batch_size=128, epochs=100, repeat=2, seed=True, diversity=False):
    
    # create a model and do a reinit function
    filename = filename+'_{}_{}_{}_{}_{}'.format(data_name, network_name, active_name, n_pool, attack)
    img_size = getSize(data_name)
    #
    active_train_attack = None 
    if active_name == 'adv_train':
        active_train_attack = attack

    model, labelled_data, unlabelled_data, test_data, full_train = loading(num_sample, network_name, data_name, seed=seed)

    percentage_data = num_sample #len(labelled_data)
    log('START')
    while( percentage_data < n_pool):
        log('percentage_data = ', percentage_data)

        model = active_training(labelled_data, network_name, img_size,
                             batch_size=batch_size, epochs=epochs, 
                             repeat=repeat, device=device, attack=active_train_attack)
    
        log("Evaluate and report test acc of model")
        evaluate(model, test_data, percentage_data, id_exp, repo, filename, device, batch_size)
        t = time.time()
        log(f"Active selection: {active_name}")    
        query, unlabelled_data = active_selection(model, unlabelled_data, nb_query, active_name, attack, device, diversity=diversity) # TO DO
        # add query to the labelled set
        labelled_data.cat(query)
        #update percentage_data
        percentage_data = len(labelled_data)
        t = time.time() - t 
        log("{}: active selection time {:.2f} seconds.".format(percentage_data, t))

    log('percentage_data = ', percentage_data)
    model = active_training(labelled_data, network_name, img_size,
                             batch_size=batch_size, epochs=epochs, 
                             repeat=repeat, device=device, attack=active_train_attack)
    log("Evaluate and report test acc of model")
    evaluate(model, test_data, percentage_data, id_exp, repo, filename, device, batch_size)
    log("END")

    log('training on random sample of same size')
    # subset_index = np.random.choice(full_train.indices, size=percentage_data, replace=False)
    # random_subset = Subset(full_train.dataset, subset_index)
    random_subset, _ = random_split(full_train, [percentage_data, len(full_train) - percentage_data],
                                    generator=torch.Generator().manual_seed(42))
    model = active_training(random_subset, network_name, img_size,
                             batch_size=batch_size, epochs=epochs, 
                             repeat=repeat, device=device, attack=None)
    log("Evaluate and report test acc of random sample of same size")
    evaluate(model, test_data, percentage_data, id_exp, repo, filename, device, batch_size)


    log('adversarial training on random sample of same size')
    model = active_training(random_subset, network_name, img_size,
                             batch_size=batch_size, epochs=epochs, 
                             repeat=repeat, device=device, attack=None)
    log("Evaluate and report test acc of random sample of same size (adv train)")
    evaluate(model, test_data, percentage_data, id_exp, repo, filename, device, batch_size)


    log('training on full data')
    model = active_training(full_train, network_name, img_size,
                             batch_size=batch_size, epochs=epochs, 
                             repeat=repeat, device=device, attack=None)
    log("Evaluate and report test acc of full model")
    evaluate(model, test_data, percentage_data, id_exp, repo, filename, device, batch_size)

    log('adversarial training on full data')
    model = active_training(full_train, network_name, img_size,
                             batch_size=batch_size, epochs=epochs, 
                             repeat=repeat, device=device, attack=attack)
    log("Evaluate and report test acc of full model (adv train)")
    evaluate(model, test_data, percentage_data, id_exp, repo, filename, device, batch_size)
        
#%%
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Active Learning')

    parser.add_argument('--id_experiment', type=int, default=4, help='id number of experiment')
    parser.add_argument('--repo', type=str, default='.', help='repository for log')
    parser.add_argument('--filename', type=str, default='acc', help='csv filename')
    parser.add_argument('--num_sample', type=int, default=100, help='size of the initial training set')
    parser.add_argument('--n_pool', type=int, default=200, help='Final train size')
    parser.add_argument('--data_name', type=str, default='mnist', help='dataset')
    parser.add_argument('--network_name', type=str, default='resnet18', help='network')
    parser.add_argument('--active', type=str, default='saaq', help='active techniques')
    parser.add_argument('--attack', type=str, default='fgsm', help='type of attack')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--repeat', type=int, default=2, help='train repeats for active training')
    parser.add_argument('--seed', type=int, default=1, help='if True(>0) the initial 100 images will be the same everytime')
    parser.add_argument('--diversity', type=int, default=0, help='if True(>0) diversity selection applied')

    args = parser.parse_args()
                                                                                                             



                                                                                                                

    id_exp = args.id_experiment
    repo=args.repo
    filename=args.filename
    if filename.split('.')[-1]=='csv':
        filename=filename.split('.csv')[0]
        
    data_name = args.data_name
    network_name = args.network_name
    active_name = args.active
    num_sample = args.num_sample
    n_pool = args.n_pool
    attack = args.attack
    batch_size=args.batch_size
    epochs = args.epochs
    repeat=args.repeat
    seed = args.seed
    diversity= args.diversity

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} device")

    start = time.time()
    
    active_learning(num_sample=num_sample,
                    data_name=data_name,
                    network_name=network_name,
                    active_name=active_name,
                    attack=attack,
                    id_exp=id_exp,
                    n_pool=n_pool,
                    repo=repo,
                    filename=filename,
                    device=device,
                    batch_size=batch_size,
                    epochs=epochs,
                    repeat=repeat,
                    seed=bool(seed),
                    diversity=bool(diversity))

    t = time.time() - start
    log('Time: {:.2f} seconds'.format(t))
    f = open("time.txt", 'a')
    f.write('Time: {:.2f} seconds'.format(t))
    f.close()