
'''

    데이터셋은 OCR 이미지를 사용하고 ROI를 80%이상 정확하게 잡으면 In-distribution 나머지 20%만 나오도록 ROI가 외곽으로 나가면 Out-distribution
    add noise 하는 부분은 Sign 함수, magnitude 대신에 weight로 대체

    training set으로 mean, covariance 계산


'''

import torch
from torchvision import transforms
from torch.utils.data import ConcatDataset

import numpy as np

import data_loader
import model_loader

import argparse

def ArgsParse():
    parser = argparse.ArgumentParser(description="MyDeep Mahalanobis Detector")
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for data loader')
    parser.add_argument('--pretrained_model', type=str, required=True, help="path to pretrained model")
    parser.add_argument('--gpu_id', type=int, default=0, help="gpu index")

    return parser.parse_args()

def main(args):

    torch.cuda.manual_seed(0)
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    # dataloader
    data_type = "cifar10" # 임시
    in_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    data_path = "./data" # 임시

    train_loader_cifar, test_loader_cifar = data_loader.getTargetDataSet(data_type=data_type,
                                                             batch_size=args.batch_size,
                                                             input_TF=in_transform,
                                                             dataroot=data_path)

    train_loader_svhn = data_loader.getNonTargetDataSet('svhn', args.batch_size, in_transform, data_path)

    train_dataset = ConcatDataset([
        train_loader_cifar.dataset,
        train_loader_svhn.dataset
    ])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    # model load
    net_type = "resnet" # 임시
    model = model_loader.model_generator(net_type=net_type, pretrained_model=args.pretrained_model, device=device)

    model.to(device=device)

    out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize']
    # for out_dist in out_dist_list:
    #     out_test_loader = data_loader.getNonTargetDataSet(out_dist, args.batch_size, in_transform, data_path)
    #
    #     for data, target in out_test_loader:
    #         data, target = data.to(device), target.to(device)
    #         print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    for epoch in range(100):
        if epoch == 99:
            print()
        # train
        loss_list = []
        mse_list = []
        kl_list = []
        for idx ,(data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            # pred, mean_list, std_list = model(data)
            pred, mean_list, logvar_list = model(data)
            # print(target, pred)
            # for i in range(5):
            #     print(f"std_{i} max-{torch.max(std_list[i])} / min-{torch.min(std_list[i])}")
            loss, mse, kl = loss_func(target, pred, mean_list, logvar_list)
            # print(f"step :{idx}", round(loss.item(),6))
            loss_list.append(loss.item())
            mse_list.append(mse.item())
            kl_list.append(kl.item())
            # loss = criterion(pred, target.type(torch.float32))
            # print(loss)
            loss.backward()
            optimizer.step()

        print(f"{epoch}/loss: ", np.mean(loss_list))
        print(f"{epoch}/mse: ", np.mean(mse_list))
        print(f"{epoch}/KL: ", np.mean(kl_list))
    # valid

def loss_func(input, output, mu_list, logvar_list):
    marginal_likelihood = torch.mean(torch.sum(input * torch.log(output) + (1 - input) * torch.log(1 - output)))
    # marginal_likelihood = torch.mean(torch.square(output - input))
    # print("mse :", marginal_likelihood.item())
    temp_loss = []
    for mu, logvar in zip(mu_list, logvar_list):
        temp_loss.append(torch.mean(-0.5 * torch.sum( (1 + logvar) - torch.square(mu) - torch.exp(logvar), dim=1)))

    kl_divergence = torch.mean(torch.Tensor(temp_loss))
    # print("KL divergence : ", kl_divergence.item())
    ELBO = marginal_likelihood - kl_divergence
    loss = -ELBO

    if loss.item() == float("inf"):
        print()
    elif loss.item() == float("-inf"):
        print()
    elif loss.item() == float("nan"):
        print()
    return loss, marginal_likelihood, kl_divergence

if __name__ == "__main__":
    args = ArgsParse()
    main(args)