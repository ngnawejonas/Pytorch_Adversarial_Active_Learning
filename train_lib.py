import time
from tqdm import tqdm as tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method        
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent


EARLY_STOPPING = True
SHOW_LOG = False

def log(msg, *args):
    if SHOW_LOG:
        print(msg, *args)

# Edit from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, current_score):

        if self.best_score is None:
            self.best_score = current_score

        elif self.best_score < current_score + self.delta: # there is no improvement
            self.counter += 1
            # log("best {:.2f}, current {:.2f}".format(self.best_score, current_score))
            # log(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0


def train(dataloader, model, loss_fn, optimizer, device, attack=None, verbose=True):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        if attack:
            X = attack_fn(model, X, attack, test=False)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0 and verbose:
            loss, current = loss.item(), batch * len(X)
            log(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# def adv_train(dataloader, model, loss_fn, optimizer, device, attack, verbose=True):
#     log('Adversarial training')
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)
#         X = attack_fn(model, X, attack, test=False)
#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if batch % 100 == 0 and verbose:
#             loss, current = loss.item(), batch * len(X)
#             log(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def simple_test(dataloader, model, loss_fn=None, device=None, verbose=True):
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    acc = 100*correct
    if verbose:
        log(f"Test Accuracy: {(acc):>0.1f}% \n")
    return acc

# def robust_test(dataloader, model, loss_fn=None, attack=None, device=None, verbose=True):
#     if loss_fn is None:
#         loss_fn = nn.CrossEntropyLoss()
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     correct = 0
#     for X, y in dataloader:
#         X, y = X.to(device), y.to(device)
#         # SHOW_LOG = True
#         log("attack...", num_batches)
#         t = time.time()
#         X = attack_fn(model, X, attack)
#         log('{:.2f} secs'.format(time.time()-t))
#         model.eval()
#         with torch.no_grad(): 
#             pred = model(X)
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     correct /= size
#     acc = 100*correct
#     if verbose:
#         log(f"Robust test Accuracy: {(acc):>0.1f}% \n")
#     # SHOW_LOG = False
#     return acc

def robust_test(dataloader, model, loss_fn=None, attack=None, device=None, verbose=True):
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    correct = 0
    model.eval()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        # SHOW_LOG = True
        # log("attack...", num_batches)
        t = time.time()
        X = attack_fn(model, X, attack)
        # log('{:.2f} secs'.format(time.time()-t))
        pred = model(X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    acc = 100*correct
    if verbose:
        log(f"Robust test Accuracy: {(acc):>0.1f}% \n")
    # SHOW_LOG = False
    return acc


def test(dataloader, model, loss_fn=None, device=None, verbose=True):
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    acc = 100*correct
    if verbose:
        log(f"Validation Error: \n Accuracy: {(acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, acc

def get_optimzer(model, network_name='resnet18', data_name='mnist'):
    if network_name=='resnet18' and data_name in ['mnist', 'fashion_mnist']:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005)
        scheduler = None
    if network_name=='resnet18' and data_name == 'cifar10':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=0.1, weight_decay=5e-4,
                                    momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return optimizer, scheduler


def train_model(train_data, validation_data, model, epochs=5, 
                batch_size=64, optz=None, device=None, attack=None):
    """ train model"""
    model = model.to(device)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size)
    loss_fn = nn.CrossEntropyLoss()
    # if optimizer is None:
    #     optimizer = get_optimzer(model)
    scheduler = optz[1]
    optimizer = optz[0]
    early_topping = EarlyStopping(patience=epochs//5 if epochs//5 > 15 else epochs)

    for t in tqdm(range(epochs), ncols=100):
        verbose =  t%10 == 0
        if verbose:
            log(f"Epoch {t+1}\n-------------------------------")

        train(train_dataloader, model, loss_fn, optimizer, device, attack=attack, verbose=verbose)

        validation_loss, acc = test(validation_dataloader, model, loss_fn, device, verbose=verbose)
        if scheduler is not None:
            scheduler.step()
        early_topping(validation_loss)
        if early_topping.early_stop:
            break
    log("End Training!")
    return validation_loss

def attack_fn(model, true_image, option='fgsm', test=True):
    eps = 0.5 if test else 0.3
    if option == 'fgsm':
        return fast_gradient_method(model_fn=model, x=true_image,
                                    eps=0.5, norm=np.inf, targeted=False,
                                    sanity_checks=False)
    elif option == 'bim':
        return projected_gradient_descent(model_fn=model, x=true_image, 
                                        eps=eps, eps_iter=1e-2, nb_iter=10, norm=np.inf,
                                        targeted=False, rand_init=False, rand_minmax=None, 
                                        sanity_checks=False)
    elif option == 'pgd':
        return projected_gradient_descent(model_fn=model, x=true_image, 
                                        eps=eps, eps_iter=1e-2, nb_iter=10, norm=np.inf,
                                        targeted=False, rand_init=True, rand_minmax=0.3, 
                                        sanity_checks=False)
    else:
        raise NotImplementedError('option "{}" not implemented option'.format(option))