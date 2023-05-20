import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def pgd_attack(model, X, y, device, attack_params, logadv=False): 
    """
        Reference: 
            https://github.com/yaodongyu/TRADES/blob/master/pgd_attack_cifar10.py 
            L2 attack: https://github.com/locuslab/robust_overfitting/blob/master/train_cifar.py
        Args: 
            model: pretrained model 
            X: input tensor
            y: input target 
            attack_params:
                loss_type: 'ce', 'kl' or 'mart'
                epsilon: attack boundary
                step_size: attack step size 
                num_steps: number attack step 
                order: norm order (norm l2 or linf)
                random_init: random starting point 
                x_min, x_max: range of data 
    """
    # model.eval()

    X_adv = Variable(X.data, requires_grad=True)

    not_targeted = 1 if not attack_params['targeted'] else -1 
    target = y if not attack_params['targeted'] else attack_params['y_target'] 

    if attack_params['random_init']:
        random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-attack_params['epsilon'], 
                                                            attack_params['epsilon']).to(device)
        X_adv = Variable(X_adv.data + random_noise, requires_grad=True)
    
    X_adves = []
    for _ in range(attack_params['num_steps']):
        opt = optim.SGD([X_adv], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            if attack_params['loss_type'] == 'ce':
                loss = nn.CrossEntropyLoss()(model(X_adv), target)
            elif attack_params['loss_type'] == 'kl': 
                loss = nn.KLDivLoss()(F.softmax(model(X_adv), dim=1), 
                                    F.softmax(model(X), dim=1))

        loss.backward()
        eta = attack_params['step_size'] * X_adv.grad.data.sign()
        X_adv = Variable(X_adv.data + not_targeted * eta, requires_grad=True)
        eta = torch.clamp(X_adv.data - X.data, 
                            -attack_params['epsilon'], 
                            attack_params['epsilon'])
        X_adv = Variable(X.data + eta, requires_grad=True)
        X_adv = Variable(torch.clamp(X_adv, 
                            attack_params['x_min'], 
                            attack_params['x_max']), requires_grad=True)

        if logadv:
            X_adves.append(X_adv)

    # switch_status(model, status)
    X_adv = Variable(X_adv.data, requires_grad=False)
    return X_adv, X_adves
