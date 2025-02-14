# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import random
import numpy as np
from collections import OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from domainbed import networks
from domainbed import clip_networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, split_meta_train_test, ParamDict,
    MovingAverage, l2_between_dicts, proj, Nonparametric
)


ALGORITHMS = [
    'ERM',
    'RES',
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches):
        x = torch.cat([x for x,y in minibatches])
        y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(x), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

    def predict_wf(self, x):
        feat = self.featurizer(x)
        pred = self.classifier(feat)
        return feat, pred

# DomainNet: 0.02
# PACS: 0.005
# VLCS: 0.05
class MEMA:
    def __init__(self, network, decay_tec=0.001, decay_stu=0.005):
        self.network = network
        self.network_ema = copy.deepcopy(network)
        self.decay_tec = decay_tec
        self.decay_stu = decay_stu

    def update_ema(self, network):
        # update teacher model parameter
        new_dict = {}
        for (name,param_q), (_,param_k) in zip(network.state_dict().items(), self.network_ema.state_dict().items()):
            new_dict[name] = self.decay_tec * param_q.data.detach().clone() \
                           + (1.0 - self.decay_tec) * param_k.data.detach().clone()
        self.network_ema.load_state_dict(new_dict)

        # update student model parameter
        new_dict = {}
        for (name,param_q), (_,param_k) in zip(network.state_dict().items(), self.network_ema.state_dict().items()):
            new_dict[name] = (1.0 - self.decay_stu) * param_q.data.detach().clone() \
                           + self.decay_stu * param_k.data.detach().clone()
        self.network.load_state_dict(new_dict)

class RS(Algorithm, MEMA):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RS, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        MEMA.__init__(self, self.network)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches])
        y = torch.cat([y for x, y in minibatches])

        aug_mode = random.choice(['none']) 
        pred = self.classifier(self.featurizer(x, self.classifier, aug_mode))
        loss = F.cross_entropy(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_ema(self.network)
        return {'loss': loss.item()}

    def predict(self, x):
        self.network_ema.eval()
        return self.network_ema(x)

    def predict_wf(self, x):
        feat = self.network_ema.featurizer(x)
        pred = self.network_ema.classifier(feat)
        return feat, pred



class RES(Algorithm, MEMA):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RES, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        MEMA.__init__(self, self.network)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches])
        y = torch.cat([y for x, y in minibatches])

        aug_mode = random.choice(['none', 'freq_dropout', 'freq_mixup', 'freq_noise']) 
        pred = self.classifier(self.featurizer(x, self.classifier, aug_mode))
        loss = F.cross_entropy(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_ema(self.network)
        return {'loss': loss.item()}

    # def predict(self, x):
    #     return self.network(x)

    def predict(self, x):
        self.network_ema.eval()
        return self.network_ema(x)


class IE(Algorithm, MEMA):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IE, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        MEMA.__init__(self, self.network)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches])
        y = torch.cat([y for x, y in minibatches])

        aug_mode = random.choice(['none', 'image_freq_dropout', \
                                  'image_freq_noise', 'image_freq_mixup']) 
        pred = self.classifier(self.featurizer(x, self.classifier, aug_mode))
        loss = F.cross_entropy(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_ema(self.network)
        return {'loss': loss.item()}

    def predict(self, x):
        self.network.eval()
        return self.network(x)


class IE_Dropout(Algorithm, MEMA):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IE_Dropout, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        MEMA.__init__(self, self.network)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches])
        y = torch.cat([y for x, y in minibatches])

        aug_mode = random.choice(['none', 'image_freq_dropout' ]) 
        pred = self.classifier(self.featurizer(x, self.classifier, aug_mode))
        loss = F.cross_entropy(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_ema(self.network)
        return {'loss': loss.item()}

    def predict(self, x):
        self.network.eval()
        return self.network(x)




class FE(Algorithm, MEMA):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(FE, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        MEMA.__init__(self, self.network)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches])
        y = torch.cat([y for x, y in minibatches])

        aug_mode = random.choice(['none', 'feat_dropout', \
                                  'feat_noise', 'feat_mixup']) 
        if aug_mode == 'feat_mixup':
            mixup_index = torch.randperm(x.size(0))
            mixup_alpha = torch.rand(x.size(0)).to(x.device)
            pred = self.classifier(self.featurizer(x, self.classifier, aug_mode, mixup_index, mixup_alpha))
            loss = mixup_alpha * F.cross_entropy(pred, y, reduction='none') \
                 + (1-mixup_alpha) * F.cross_entropy(pred, y[mixup_index], reduction='none')
            loss = loss.mean()
        else:
            pred = self.classifier(self.featurizer(x, self.classifier, aug_mode))
            loss = F.cross_entropy(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_ema(self.network)
        return {'loss': loss.item()}

    def predict(self, x):
        self.network.eval()
        return self.network(x)


class FE_Dropout(Algorithm, MEMA):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(FE_Dropout, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        MEMA.__init__(self, self.network)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches])
        y = torch.cat([y for x, y in minibatches])

        aug_mode = random.choice(['none', 'feat_dropout']) 
        pred = self.classifier(self.featurizer(x, self.classifier, aug_mode))
        loss = F.cross_entropy(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_ema(self.network)
        return {'loss': loss.item()}

    def predict(self, x):
        self.network.eval()
        return self.network(x)


class FE_Noise(Algorithm, MEMA):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(FE_Noise, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        MEMA.__init__(self, self.network)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches])
        y = torch.cat([y for x, y in minibatches])

        aug_mode = random.choice(['none', 'feat_noise']) 
        pred = self.classifier(self.featurizer(x, self.classifier, aug_mode))
        loss = F.cross_entropy(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_ema(self.network)
        return {'loss': loss.item()}

    def predict(self, x):
        self.network.eval()
        return self.network(x)


class FE_Mixup(Algorithm, MEMA):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(FE_Mixup, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        MEMA.__init__(self, self.network)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches])
        y = torch.cat([y for x, y in minibatches])
        aug_mode = random.choice(['none', 'feat_mixup']) 

        if aug_mode == 'feat_mixup':
            mixup_index = torch.randperm(x.size(0))
            mixup_alpha = torch.rand(x.size(0)).to(x.device)
            pred = self.classifier(self.featurizer(x, self.classifier, aug_mode, mixup_index, mixup_alpha))
            loss = mixup_alpha * F.cross_entropy(pred, y, reduction='none') \
                 + (1-mixup_alpha) * F.cross_entropy(pred, y[mixup_index], reduction='none')
            loss = loss.mean()
        else:
            pred = self.classifier(self.featurizer(x, self.classifier, aug_mode))
            loss = F.cross_entropy(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_ema(self.network)
        return {'loss': loss.item()}

    def predict(self, x):
        self.network.eval()
        return self.network(x)




