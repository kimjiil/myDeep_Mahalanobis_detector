
'''

    데이터셋은 OCR 이미지를 사용하고 ROI를 80%이상 정확하게 잡으면 In-distribution 나머지 20%만 나오도록 ROI가 외곽으로 나가면 Out-distribution
    add noise 하는 부분은 Sign 함수, magnitude 대신에 weight로 대체

    training set으로 mean, covariance 계산


'''

import torch
from torchvision import transforms

import data_loader
import model_loader

import argparse

def ArgsParse():
    parser = argparse.ArgumentParser(description="MyDeep Mahalanobis Detector")
    parser.add_argument('--batch_size', type=int, default=5, help='batch size for data loader')
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

    train_loader, test_loader = data_loader.getTargetDataSet(data_type=data_type,
                                                             batch_size=args.batch_size,
                                                             input_TF=in_transform,
                                                             dataroot=data_path)

    # model load
    net_type = "resnet" # 임시
    model = model_loader.model_generator(net_type=net_type, pretrained_model=args.pretrained_model, device=device)

    model.to(device=device)


    # train
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        model(data)

    # valid


if __name__ == "__main__":
    args = ArgsParse()
    main(args)