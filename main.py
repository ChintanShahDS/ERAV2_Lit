'''Train CIFAR10 with PyTorch.'''
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelSummary

# from torchsummary import summary
# from tqdm import tqdm

import sys
sys.path.insert(0, '/content/ERAV2_Main')

from models import *
from utils import *

# Fast AI method
def find_maxlr(mymodel, train_loader):
    from torch_lr_finder import LRFinder
    criterion = nn.CrossEntropyLoss()
    optimizer_lr = optim.SGD(mymodel.parameters(), lr=1e-6, weight_decay=1e-1)
    # optimizer_lr = optim.Adam(mymodel.parameters(), lr=1e-7, weight_decay=1e-1)
    lr_finder = LRFinder(mymodel, optimizer_lr, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=10, num_iter=200)
    __, maxlr = lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state
    print("max_LR:", maxlr)
    return maxlr

def runTraining(train_loader, test_loader, initialModelPath, optimizer_name, criterion_name, scheduler_name, device, num_epochs=20, base_lr=0.01):

    initialModelPath = 'InitialModel.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    AVAIL_GPUS = min(1, torch.cuda.device_count())

    # Model
    print('==> Building model..')
    mymodel = LitResNet18(data_dir='../data', num_classes=10, learning_rate=0.01)
    mymodel = mymodel.to(device)
    torch.save(mymodel, initialModelPath)
    ds_mean = (0.4914, 0.4822, 0.4465)
    ds_std = (0.247, 0.243, 0.261)
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(ds_mean, ds_std),
        ])
    kwargs = {'batch_size':batchsize, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
    
    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, **kwargs)

    maxlr = find_maxlr(mymodel, train_loader)

    mymodel = LitResNet18(data_dir='../data', num_classes=10, learning_rate=0.01, max_lr=maxlr)
    
    misclassified_collector = MisclassifiedCollector()
                
    trainer = Trainer(
        log_every_n_steps=1, 
        # enable_model_summary=True,
        max_epochs = 3,
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        gpus = AVAIL_GPUS,
        enable_progress_bar=True,
        logger=CSVLogger(save_dir="logs/"),
        callbacks=[ModelSummary(max_depth=-1), LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10), misclassified_collector],
    )
    trainer.fit(mymodel)
    print('==> End of the training and results..')
    
    trainer.test()
    print('==> End of the testing and results..')

    df_misclassified = pd.DataFrame(misclassified_collector.misclassified_data)
    print(df_misclassified.sample(n=10))

    showIncorrectPreds(numImages, df_misclassified, classes)
    mymodel, optimizer, criterion, scheduler = setupTrainingParams(initialModelPath, optimizer_name, criterion_name, scheduler_name, train_loader, num_epochs, base_lr)
    # Data to plot accuracy and loss graphs
        
    return mymodel, train_losses, test_losses, train_accs, test_accs
    
# This is to be properly created
# Currently a temporary thing - Able to run as a Python main earlier
# But need to create such a way that can be run as functions or as main
def main(batchsize=256, num_epochs=20, base_lr=0.01, optimizer_name='SGD', criterion_name='NLL', scheduler_name='OneCycleLR'):

    initialModelPath = 'InitialModel.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model
    print('==> Building model..')
    mymodel = LitResNet18(data_dir='../data', num_classes=10, learning_rate=0.01)
    mymodel = mymodel.to(device)
    torch.save(mymodel, initialModelPath)
    ds_mean = (0.4914, 0.4822, 0.4465)
    ds_std = (0.247, 0.243, 0.261)
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(ds_mean, ds_std),
        ])
    kwargs = {'batch_size':batchsize, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
    
    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, **kwargs)

    maxlr = find_maxlr(mymodel, train_loader)

    mymodel = LitResNet18(data_dir='../data', num_classes=10, learning_rate=0.01, max_lr=maxlr)
    trainer = Trainer(
        max_epochs = 3,
        enable_progress_bar=True
    )
    trainer.fit(mymodel)
    print('==> End of the training and results..')
    
    trainer.test()
    print('==> End of the testing and results..')

    # print('==> Training model..')
    # mymodel, train_losses, test_losses, train_accs, test_accs = runTraining(
        # train_loader, test_loader, initialModelPath, optimizer_name, 
        # criterion_name, scheduler_name, device, num_epochs=num_epochs, base_lr=base_lr)

    # print('==> Accuracy plots..')
    # drawLossAccuracyPlots(train_losses, train_accs, test_losses, test_accs)

    # print('==> Incorrect outcomes..')
    # numImages = 10
    # images, nonMatchingLabels, incorrectPreds = incorrectOutcomes(mymodel, device, test_loader, numImages)
    # showIncorrectPreds(numImages, images, incorrectPreds, nonMatchingLabels,classes)

    # print('==> Incorrect outcomes explanation using GradCam..')
    # showGradCam(numImages, images, incorrectPreds, nonMatchingLabels, classes, mymodel, [mymodel.layer4[-1]])
    

if __name__ == '__main__':
    print("Start of the main module")
    # parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    # parser.add_argument('--num_epochs', default=20, type=int, help='number of epochs')
    # parser.add_argument('--batchsize', default=20, type=int, help='Batch size')
    # parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    # parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer')
    # parser.add_argument('--criterion', default='CrossEntropyLoss', type=str, help='loss criteria')
    # parser.add_argument('--lrscheduler', default='OneCycleLR', type=str, help='lr scheduler')
    # # parser.add_argument('--resume', '-r', action='store_true',
                        # # help='resume from checkpoint')
    # args = parser.parse_args()
    # print("Arguments parsing complete")

    # # getting all arguments
    # num_epochs = args.num_epochs
    # batchsize = args.batchsize
    # base_lr = args.lr
    # optimizer_name = args.optimizer
    # criterion_name = args.criterion
    # scheduler_name = args.lrscheduler
    
    batchsize = 256
    num_epochs = 20
    
    main()
