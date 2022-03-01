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
from torch.utils.data import DataLoader, random_split, Subset

from contextlib import closing
from build_model import build_model_func
from build_data import build_data_func, getSize, MyDataSet
from train_lib import train_model, test
from adversarial_active_criterion import Adversarial_DeepFool



#%%
def active_training(labelled_data, network_name, img_size,
                    batch_size=64, epochs=20, repeat=2, device=None):
    # split into train and validation
    
    N = len(labelled_data)
    n_train = (int) (N*0.8)

    batch_train = min(batch_size, n_train)
    steps_per_epoch = int(n_train/batch_train) + 1
    best_model = None
    best_loss = np.inf
    for i in range(repeat):
        print(f'training No {i+1}')
        # shuffle data and split train and val
        train_data, val_data = random_split(labelled_data, [n_train, N - n_train],
                            generator=torch.Generator().manual_seed(42))
     
        model = build_model_func(network_name, img_size)

        loss = train_model(train_data, val_data, model, epochs, batch_size, device)
                                   
        if loss < best_loss:
            best_loss = loss;
            best_model = model

    del model
    del loss

    return best_model

#%%
def evaluate(model, test_data, percentage, id_exp, repo, filename, device):
    t = time.time()
    test_dataloader = DataLoader(test_data)
    loss, acc = test(test_dataloader, model.to(device), device=device)
    t = time.time() - t
    print("eval time: {:.2f}".format(t))
    with closing(open(os.path.join(repo, filename), 'a')) as csvfile:
        # TO DO
        writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([str(id_exp), str(percentage), str(acc)])
         

#%%
def get_weights(model):
    layers = model.layers
    weights=[]
    for layer in layers:
        if layer.trainable_weights:
            weights_layer = layer.trainable_weights
            ##### The elem.get_value() function does not work in Tensorflow 2 so I change it by K.get_value(elem)
            weights+=[K.get_value(elem) for elem in weights_layer]
    return weights
    
def load_weights(model, weights):
    layers = model.layers
    index=0
    for layer in layers:
        if layer.trainable_weights:
            weights_layer = layer.trainable_weights
            for elem in weights_layer:
                ##### The elem.set_value() function does not work with Tensorflow 2 so I change it by K.set_value(elem)
                K.set_value(elem,weights[index])
                index+=1
    return model
                
                
def loading(repo, filename, num_sample, network_name, data_name):
    # check if file exists
    img_size = getSize(data_name) # TO DO
    model=build_model_func(network_name, img_size)
    filename = filename.split('.pkl')
    f_weights = filename[0]+'_weights.pkl'
    f_l_data = filename[0]+'_labelled.pkl'
    f_u_data = filename[0]+'_unlabelled.pkl'
    f_t_data = filename[0]+'_test.pkl'
    if (os.path.isfile(os.path.join(repo, f_weights)) and \
        os.path.isfile(os.path.join(repo, f_l_data)) and \
        os.path.isfile(os.path.join(repo, f_u_data)) and \
        os.path.isfile(os.path.join(repo, f_t_data))):
        
        
        
        with closing(open(os.path.join(repo, f_weights), 'rb')) as f:
            weights = pickle.load(f)
            model = load_weights(model, weights)
            
        with closing(open(os.path.join(repo, f_l_data), 'rb')) as f:
            labelled_data = pickle.load(f)   
            
        with closing(open(os.path.join(repo, f_u_data), 'rb')) as f:
            unlabelled_data = pickle.load(f) 
            
        with closing(open(os.path.join(repo, f_t_data), 'rb')) as f:
            test_data = pickle.load(f)
    else:

        labelled_data, unlabelled_data, test_data = build_data_func(data_name, num_sample=num_sample)
    
    return model, labelled_data, unlabelled_data, test_data


def active_selection(model, unlabelled_data, nb_data, active_method, attack, repo, tmp_adv, device):
    assert active_method in ['uncertainty', 'random', 'aaq', 'saaq'], ('Unknown active criterion %s', active_method)
    if active_method=='uncertainty':
        query, unlabelled_data = uncertainty_selection(model, unlabelled_data, nb_data)
    if active_method=='random':
        query, unlabelled_data = random_selection(unlabelled_data, nb_data)
    if active_method=='aaq':
        query, unlabelled_data = adversarial_selection(model, unlabelled_data, nb_data, attack, False, repo, tmp_adv, device)
    if active_method=='saaq':
        query, unlabelled_data = adversarial_selection(model, unlabelled_data, nb_data, attack, True, repo, tmp_adv, device)       
    return query, unlabelled_data
    
def random_selection(unlabelled_data, nb_data):
    index = np.random.permutation(len(unlabelled_data[0]))
    index_query = index[:nb_data]
    index_unlabelled = index[nb_data:]
    
    return (unlabelled_data[0][index_query], unlabelled_data[1][index_query]), \
           (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled])
           
def uncertainty_selection(model, unlabelled_data, nb_data):

    preds = model.predict(unlabelled_data[0])
    log_pred = -np.log(preds)
    entropy = np.sum(preds*log_pred, axis=1)
    # do entropy
    index = np.argsort(entropy)[::-1]
    
    index_query = index[:nb_data]
    index_unlabelled = index[nb_data:]

    new_data = unlabelled_data[0][index_query]
    new_labels = unlabelled_data[1][index_query]
    """
    else:
        new_data = np.concatenate([labelled_data[0], unlabelled_data[0][index_query]], axis=0)
        new_labels = np.concatenate([labelled_data[1], unlabelled_data[1][index_query]], axis=0)
    """
    return (new_data, new_labels), \
           (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled])
                 
def adversarial_selection(model, unlabelled_data, nb_data, attack='fgsm', add_adv=False, repo='.', filename = None, device=None):

    n_channels, img_nrows, img_ncols, nb_classes  = 1, 28, 28, 10

    active = Adversarial_DeepFool(model=model, n_channels=n_channels,
                                  img_nrows=img_nrows, img_ncols=img_ncols, nb_class=nb_classes, device=device)
    # select a subset of size 10*nb_data
    u_size = len(unlabelled_data.indices)
    n = min(300, u_size)
    subset_index = np.random.choice(unlabelled_data.indices, size=n, replace=False)
    # print(n, subset_index)
    subset = Subset(unlabelled_data.dataset, subset_index)

    chosen_indices, attacked_images = active.generate(subset, attack, diversity=True)
    
    # remove chosen indexes from unlabelled_data
    for idx in chosen_indices:
        unlabelled_data.indices.remove(idx)
    
    # get selected images
    subset.indices  = chosen_indices
    dtl = DataLoader(subset, batch_size=len(subset))
    for X,y in dtl:
        new_data = MyDataSet(data=X, targets=y)

    if add_adv:
        new_data.add_data(attacked_images, unlabelled_data.dataset.targets[chosen_indices])

    return new_data, unlabelled_data
               

#%%
def active_learning(num_sample, data_name, network_name, active_name, attack='fgsm',
                    id_exp=0, nb_query=100, n_pool = 2000, repo='test', filename='test.csv', device=None):
    
    # create a model and do a reinit function
    tmp_filename = 'tmp_{}_{}_{}.pkl'.format(data_name, network_name, active_name)
    tmp_adv = None
    if active_name in ['aaq', 'saaq']:
        tmp_adv = 'adv_{}_{}_{}'.format(data_name, network_name, active_name)
    filename = filename+'_{}_{}_{}_{}_{}'.format(data_name, network_name, active_name, n_pool, attack)
    img_size = getSize(data_name)
    # TO DO filename
    
    model, labelled_data, unlabelled_data, test_data = loading(repo, tmp_filename, num_sample, network_name, data_name)

    batch_size = 128
    percentage_data = num_sample #len(labelled_data)

    print('START')
    while( percentage_data < n_pool):
        print('percentage_data = ', percentage_data)
        model = active_training(labelled_data, network_name, img_size, batch_size=batch_size, epochs=50, device=device)
    
        print("Evaluate and report test acc of model")
        evaluate(model, test_data, percentage_data, id_exp, repo, filename, device)
        t = time.time()
        print("Active selection")    
        query, unlabelled_data = active_selection(model, unlabelled_data, nb_query, active_name, attack, repo, tmp_adv, device) # TO DO
        # add query to the labelled set
        labelled_data.cat(query)
        #update percentage_data
        percentage_data +=nb_query
        t = time.time() - t 
        print("active selection time {:.2f} seconds.".format(t))

    print('percentage_data = ', percentage_data)
    model = active_training(labelled_data, network_name, img_size, batch_size=batch_size, epochs=50)
    print("Evaluate and report test acc of model")
    evaluate(model, test_data, percentage_data, id_exp, repo, filename, device)
    print("END")
        
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

    args = parser.parse_args()
                                                                                                             



                                                                                                                

    id_exp = args.id_experiment
    repo=args.repo
    filename=args.filename
    if filename.split('.')[-1]=='csv':
        filename=filename.split('.csv')[0]
        
    data_name = args.data_name
    network_name = args.network_name
    active_option = args.active
    num_sample = args.num_sample
    n_pool = args.n_pool
    attack = args.attack

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    start = time.time()
    
    active_learning(num_sample=num_sample,
                    data_name=data_name,
                    network_name=network_name,
                    active_name=active_option,
                    attack=attack,
                    id_exp=id_exp,
                    n_pool=n_pool,
                    repo=repo,
                    filename=filename,
                    device=device)

    t = time.time() - start
    print('Time: {:.2f} seconds'.format(t))
    f = open("time.txt", 'a')
    f.write('Time: {:.2f} seconds'.format(t))
    f.close()