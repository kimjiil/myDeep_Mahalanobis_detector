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

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=5, out_features=1)
            # ,torch.nn.Sigmoid()
        )

    def reparametarize(self, mu, co):
        pass

    def forward(self, x):
        batch_size = x.size(0)
        num_class = 10
        noised_img = x.repeat([1, 5, 1, 1]) + self.noise_layer(x)
        feature_list = []
        feature_out = []
        feature_dim = [64, 64, 128, 256, 512]

        mean_list = []
        std_list = []

        score_list = None
        for layer_idx, f_dim in enumerate(feature_dim):
            out_feature = self.pretrained_model.intermediate_forward(noised_img[:, 3*layer_idx:3*(layer_idx+1), :, :], layer_idx)
            out_feature = out_feature.view(out_feature.size(0), out_feature.size(1), -1)

            feature_list.append(out_feature)

            x_mean = self.mu_layer_list[layer_idx](out_feature).transpose(1, 2).contiguous()
            x_std = self.std_layer_list[layer_idx](out_feature).transpose(1, 2)
            x_variance = x_std.unsqueeze(-1) * torch.eye(f_dim).unsqueeze(0).unsqueeze(0).cuda()
            x_variance = torch.square(x_variance)
            dist = multivariate_normal.MultivariateNormal(loc=x_mean, scale_tril=x_variance)

            x_sample = dist.sample()
            x_sample = x_sample - x_mean
            temp = torch.bmm(x_sample.unsqueeze(2).view(-1, 1, f_dim), torch.inverse(x_variance).view(-1, f_dim, f_dim))
            _mahalanobis_score = -0.5 * torch.bmm(temp, x_sample.unsqueeze(-1).view(-1, f_dim, 1))
            _mahalanobis_score = _mahalanobis_score.view(batch_size, num_class)
            mahalanobis_score, max_idx = torch.max(_mahalanobis_score, dim=1)
            mahalanobis_score = mahalanobis_score.unsqueeze(-1)
            if layer_idx == 0:
                score_list = mahalanobis_score
            else:
                score_list = torch.cat((score_list, mahalanobis_score), dim=1)
            gather_idx = torch.Tensor([1] * f_dim).type(torch.int64).unsqueeze(0).unsqueeze(0).cuda() * max_idx.unsqueeze(-1).unsqueeze(-1)
            temp_mean = torch.gather(x_mean, 1, gather_idx).squeeze()
            temp_std = torch.gather(x_std, 1, gather_idx).squeeze()

            mean_list.append(temp_mean)
            std_list.append(temp_std)

        out = self.classifier(score_list)
        return out.squeeze(-1), mean_list, std_list

