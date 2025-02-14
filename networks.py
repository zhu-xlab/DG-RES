# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torch.fft as fft

from domainbed.lib import wide_resnet
import random
import copy


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048
        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def pooling(self, x):
        return self.network.avgpool(x).view(x.size(0), -1)

    def forward(self, x, classifier=None, aug_mode='none', mixup_index='none', mixup_alpha=0.5):
        if aug_mode=='image_freq_noise':
            x = self.freq_noise(x, std=0.75)
        if aug_mode=='image_freq_dropout':
            x = self.freq_dropout(x, p=0.5)
        if aug_mode=='image_freq_mixup':
            x = self.freq_mixup(x, min_value=0.05) 
        
        x = self.network.conv1(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)
        aug_idx = random.choice([0,1,2,3]) 
        layers = [self.network.layer1, self.network.layer2, \
                  self.network.layer3, self.network.layer4]
        for idx, layer in enumerate(layers):
            x = layer(x)
            if aug_idx==idx and aug_mode=='freq_noise':
                x = self.freq_noise(x, std=0.75)
            if aug_idx==idx and aug_mode=='freq_dropout':
                x = self.freq_dropout(x, p=0.50)
            if aug_idx==idx and aug_mode=='freq_mixup':
                x = self.freq_mixup(x, min_value=0.05) 
            if aug_idx==idx and aug_mode=='feat_noise':
                x = self.feat_noise(x, std=0.75)
            if aug_idx==idx and aug_mode=='feat_dropout':
                x = self.feat_dropout(x, p=0.50)
            if aug_idx==idx and aug_mode=='feat_mixup':
                x = self.feat_mixup(x, min_value=0.05, mixup_index=mixup_index, mixup_alpha=mixup_alpha) 
            # if aug_idx==idx and aug_mode=='spatial_dropout':
            #     res_layers = [layers[i] for i in range(idx+1, len(layers))]
            #     x = self.spatial_dropout(res_layers, classifier, x) 
        x = self.pooling(x) 
        return x   

    # def spatial_dropout(self, res_layers, classifier, x, p=0.5):
    #     # ger final features
    #     x_bar = x.clone().detach()
    #     b, c, h, w = x_bar.size()
    #     if len(res_layers) > 0:
    #         with torch.no_grad():
    #             for idx, layer in enumerate(res_layers):
    #                 x_bar = layer(x_bar)

    #     # generate attention map
    #     fc_weights = classifier.weight.data  
    #     conv_weights = fc_weights.view(fc_weights.size(0), fc_weights.size(1), 1, 1)  
    #     logit = F.conv2d(x_bar, conv_weights)  
    #     b, c, h, w = logit.size()

    #     probabilities = F.softmax(logit, dim=1)
    #     norm_logit = (-probabilities * torch.log2(probabilities + 1e-12)).sum(dim=1)

    #     norm_attn = norm_logit.view(b, h*w)
    #     logit_max  = norm_attn.max(dim=-1)[0].unsqueeze(dim=-1)
    #     logit_min  = norm_attn.min(dim=-1)[0].unsqueeze(dim=-1)
    #     norm_attn = (norm_attn - logit_min) / (logit_max - logit_min)
    #     norm_attn = norm_attn.view(b, h, w).unsqueeze(dim=1)
    #     norm_attn = F.interpolate(norm_attn, size=(7,7), mode='bilinear')

    #     # generate mask
    #     mask = torch.rand(norm_attn.size()).to(x.device).detach() * norm_attn
    #     mask = (mask < (1-p)).float()
    #     mask = F.interpolate(mask, size=x.size()[-2:], mode='bilinear')
    #     return x*mask

    def fft(self, x, is_shift=False):
        spectrum = fft.fft2(x, dim=(-2, -1))  # 在空间维度上执行2D傅里叶变换
        phase = torch.angle(spectrum)
        magnitude = torch.abs(spectrum)
        if is_shift:
            magnitude = fft.ifftshift(magnitude)
        return phase, magnitude     

    def ifft(self, magnitude, phase):
        reconstructed_spectrum = magnitude * torch.exp(1j * phase)
        reconstructed_x = fft.ifft2(reconstructed_spectrum, dim=(-2, -1)).real
        return reconstructed_x   

    def freq_noise(self, x, std=0.75):
        # extract pahse and manigtude from images by DCT
        phase, magnitude = self.fft(x)

        # enhance: noising
        b, c, h, w = x.size()
        white_noise = torch.randn((b, 1, h, w)).to(x.device).detach()
        scaled_white_noise = (1 + std * white_noise).clip(min=0, max=2)
        noised_magnitude = scaled_white_noise*magnitude

        # reconstruct images
        reconstructed_x = self.ifft(noised_magnitude, phase)
        return reconstructed_x

    def freq_dropout(self, x, p=0.50):
        # extract pahse and manigtude from images by DCT
        phase, magnitude = self.fft(x)

        # enhance: dropout
        b, c, h, w = x.size()
        mask = torch.rand((b, 1, h, w)).to(x.device)
        mask = (mask > p).float().detach()
        # mask[:,:,0,0] *= 0.5    
        mask[:,:,0,0] = 1    
        dropped_magnitude = mask*magnitude

        # reconstruct images
        reconstructed_x = self.ifft(dropped_magnitude, phase)
        return reconstructed_x

    def freq_mixup(self, x, min_value=0):
        # extract pahse and manigtude from images by DCT
        phase, magnitude = self.fft(x)

        # enhance: magnitude mixup
        b, c, h, w = x.size()
        lam = torch.rand(b).to(x.device).detach().clip(min=min_value)\
            .unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        index = torch.randperm(b)
        mixed_magnitude = lam * magnitude + (1-lam) * magnitude[index]

        # reconstruct images
        reconstructed_x = self.ifft(mixed_magnitude, phase)
        return reconstructed_x

    def feat_noise(self, x, std=0.75):
        # enhance: noising
        b, c, h, w = x.size()
        white_noise = torch.randn((b, 1, h, w)).to(x.device).detach()
        scaled_white_noise = (1 + std * white_noise).clip(min=0, max=2)
        noised_x = scaled_white_noise*x
        return noised_x

    def feat_dropout(self, x, p=0.50):
        # enhance: dropout
        b, c, h, w = x.size()
        mask = torch.rand((b, 1, h, w)).to(x.device)
        mask = (mask > p).float().detach()
        dropped_x = mask*x
        return dropped_x

    def feat_mixup(self, x, min_value=0, mixup_index='none', mixup_alpha=0.5):
        # enhance: magnitude mixup
        b, c, h, w = x.size()
        mixup_alpha = mixup_alpha.detach().unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        mixed_x = mixup_alpha * x + (1-mixup_alpha) * x[mixup_index]
        return mixed_x

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def  Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)
