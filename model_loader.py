import torch
import models
from torch.distributions import multivariate_normal
import numpy as np

def pretrained_model_load(net_type, pretrained_model, device):

    if net_type == 'resnet':
        model = models.ResNet34(num_c=10)
        model.load_state_dict(torch.load(pretrained_model, map_location=device))


    return model


def model_generator(net_type, pretrained_model, device):
    pretrained_model = pretrained_model_load(net_type, pretrained_model, device)

    ood_model = OutOfDistributionDetector(pretrained_model=pretrained_model)

    return ood_model


class OutOfDistributionDetector(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(OutOfDistributionDetector, self).__init__()
        self.pretrained_model = pretrained_model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.noise_layer = torch.nn.Conv2d(in_channels=3, out_channels=15, kernel_size=3, stride=1, padding="same")

        self.mu_layer_list = torch.nn.ModuleList([torch.nn.Linear(in_features=in_dim, out_features=10) for in_dim in [1024, 1024, 256, 64, 16]])
        # self.mu_layer_0 = torch.nn.Linear(in_features=1024, out_features=10) # N,64,1024 -> N,64,10
        # self.mu_layer_1 = torch.nn.Linear(in_features=1024, out_features=10)
        # self.mu_layer_2 = torch.nn.Linear(in_features=256, out_features=10)
        # self.mu_layer_3 = torch.nn.Linear(in_features=64, out_features=10)
        # self.mu_layer_4 = torch.nn.Linear(in_features=16, out_features=10)

        self.std_layer_list = torch.nn.ModuleList([torch.nn.Linear(in_features=in_dim, out_features=10) for in_dim in [1024, 1024, 256, 64, 16]])
        # self.std_layer_list = torch.nn.ModuleList([torch.nn.Linear(in_features=10, out_features=feature_dim) for feature_dim in [64, 64, 128, 256, 512]])
        # self.std_layer_0 = torch.nn.Linear(in_features=10, out_features=64) # N 64 1024
        # self.std_layer_1 = torch.nn.Linear(in_features=10, out_features=64) # N 64 1024
        # self.std_layer_2 = torch.nn.Linear(in_features=10, out_features=128) # N 128 256
        # self.std_layer_3 = torch.nn.Linear(in_features=10, out_features=256) # N 256 64
        # self.std_layer_4 = torch.nn.Linear(in_features=10, out_features=512) # N 512 16

    def reparametarize(self, mu, co):
        pass

    def forward(self, x):
        noised_img = self.noise_layer(x)
        feature_list = []
        feature_out = []
        for layer_idx in range(5):
            out_feature = self.pretrained_model.intermediate_forward(noised_img[:, 3*layer_idx:3*(layer_idx+1), :, :], layer_idx)
            out_feature = out_feature.view(out_feature.size(0), out_feature.size(1), -1)

            feature_list.append(out_feature)

            x_mean = self.mu_layer_list[layer_idx](out_feature).transpose(1, 2).contiguous()
            x_std = self.std_layer_list[layer_idx](out_feature).transpose(1, 2)
            x_variance = x_std * torch.eye(64).unsqueeze([0, 1]).cuda()
            random_covariance = torch.randn((64, 64)) * torch.eye(64)
            dist = multivariate_normal.MultivariateNormal(loc=x_mean.transpose(1, 2).contiguous(), scale_tril=random_covariance.cuda())

            print()
        output = None
        return output

class AddLayer(torch.nn.Module):
    def __init__(self, channel, height, width):
        super(AddLayer, self).__init__()
        self._weight = torch.nn.Parameter(torch.randn(channel, height, width))

    def forward(self, input):
        output = torch.add(input, self._weight)
        return output

