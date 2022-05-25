# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.

author : mducoffe

Step 1 : deep fool as an active learning criterion
"""

##### These comments are from Julien Choukroun
##### Now we use tensorflow.keras instead of keras because we are in Tensorflow 2

import numpy as np
import random
import torch
from torch.utils.data import DataLoader, random_split
##### I use the cleverhans functions for the attacks
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method        
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent


SHOW_LOG = False

def log(msg, *args):
    if SHOW_LOG:
        print(msg, *args)

class Adversarial_example(object):
    
    def __init__(self, model, device=None):
        
        self.model = model.to(device)
        self.device = device

    def predict(self,image):
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(image)
            y_pred = prediction.argmax(1)
        return y_pred
        
    def generate(self, data):
        raise NotImplementedError()
                
    def generate_sample(self, true_image):
        raise NotImplementedError()

    def generate_sample_adv(self, index, true_image, attack_fn, **args):
        true_image = true_image.to(self.device)
        true_label = self.predict(true_image)
        ##### I save the labels in a txt file
        file = open('labels.txt', 'a')
        file.write(str(true_label))
        file.write(",     ")
        x_i = true_image.detach().clone()
        i=0
        while self.predict(x_i) == true_label and i<10:
            x_i = attack_fn(model_fn=self.model, x=x_i, y = true_label, **args)
            i+=1

        adv_label = self.predict(x_i)
        ##### I save the adversarial's labels in a txt file
        file.write(str(adv_label))
        file.write("\n")
        file.close()
        ### perturbation
        distance = torch.norm(x_i - true_image)
        # assert distance == 0
        ##### I save the distance in a txt file
        file = open('distance.txt', 'a')
        file.write(str(index))
        file.write(",   ")
        file.write(str(distance.detach().cpu().numpy()))
        file.write("\n")
        file.close()
        return distance, x_i


class Adversarial_DeepFool(Adversarial_example):
    
    def __init__(self,  **kwargs):
        super(Adversarial_DeepFool, self).__init__(**kwargs)


        
    def generate(self, data, option='fgsm', diversity=False):

        method = ['fgsm', 'bim', 'pgd', 'mim']
        if option == 'random':
            option = random.choice(method)    
            ##### I save the option's name in a txt file
            file = open('option.txt', 'a')
            file.write(str(option))
            file.write("\n")
            file.close()
        perturbations = []
        adv_attacks = []
        dataloader = DataLoader(data)
        for i, (image, _) in enumerate(dataloader):
            # if i%10==0:
            #     log(i)
            r_i, adv_image = self.generate_sample(image, option=option, index=i)
            perturbations.append(r_i.detach())
            adv_attacks.append(adv_image.detach()[0])
            print('adv added', adv_attacks[-1].shape, adv_attacks[-1].cpu().numpy().min(), adv_attacks[-1].cpu().numpy().max())
 
        perturbations = torch.Tensor(perturbations)
        index_perturbation = perturbations.argsort()
        adv = torch.stack(adv_attacks)
        sortedAdv = adv[index_perturbation]


        if diversity:
            log('diversity selection')
            dist = []
            for i in range(len(data)):
                for j in range(i, len(data)):
                    adv_dist = torch.norm(sortedAdv[i]-sortedAdv[j])
                    dist.append(adv_dist.cpu().numpy())

            median_dist = np.median(np.unique(dist))

            new_index_perturbation = []

            for i in range(len(data)):
                index_max = np.argmax(dist[len(data)*i:len(data)*(i+1)])
                max_dist = dist[(len(data)*i)+index_max]
                if max_dist > median_dist:
                    new_index_perturbation.append(index_perturbation[i])

            chosen_indices = [data.indices[i] for i in new_index_perturbation]
            return chosen_indices, adv[new_index_perturbation]
    
        # select the chosen indices
        chosen_indices = [data.indices[i] for i in index_perturbation]        
        return chosen_indices, sortedAdv

         
    def generate_sample(self, true_image, option='fgsm', index=None):

        if option == 'fgsm':
            return self.generate_sample_adv(index, true_image, fast_gradient_method,
                                            eps=0.5, norm=np.inf, targeted=False,
                                            sanity_checks=False)
        elif option == 'bim':
            return self.generate_sample_adv(index, true_image, projected_gradient_descent,
                                            eps=0.3, eps_iter=1e-2, nb_iter=10, norm=np.inf, 
                                            targeted=False, rand_init=False, rand_minmax=None,
                                            sanity_checks=False)
        elif option == 'pgd':
            return self.generate_sample_adv(index, true_image, projected_gradient_descent,
                                            eps=0.3, eps_iter=1e-2, nb_iter=10, norm=np.inf,
                                            targeted=False, rand_init=True, rand_minmax=0.3, 
                                            sanity_checks=False)
        else:
            raise NotImplementedError('option "{}" not implemented option'.format(option))