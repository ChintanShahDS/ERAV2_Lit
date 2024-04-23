import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms.v2 as transforms
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import Callback

import os

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

class LitResNet18(pl.LightningModule):
    def __init__(self, data_dir, num_classes=10, learning_rate=0.01, max_lr=1.45E-03):
        super(LitResNet18, self).__init__()
        self.in_planes = 64
        self.data_dir = data_dir

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear = nn.Linear(512*BasicBlock.expansion, num_classes)

        self.learning_rate = learning_rate
        self.max_lr = max_lr
        self.num_classes = num_classes
        self.steps_per_epoch = 50000 / BATCH_SIZE
        self.ds_mean = (0.4914, 0.4822, 0.4465)
        self.ds_std = (0.247, 0.243, 0.261)

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        self.train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Pad(16, self.ds_mean, 'constant'),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(self.ds_mean, self.ds_std),
            transforms.RandomErasing(scale=(0.125, 0.125), ratio=(1, 1), value=self.ds_mean, inplace=False),
            transforms.CenterCrop(32),
            ])
            
        # Test data transformations
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(self.ds_mean, self.ds_std),
            ])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        
    def configure_optimizers(self):
        pct_start = 0.3
        base_momentum = 0.85
        max_momentum = 0.9
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        steps_per_epoch = int(self.trainer.estimated_stepping_batches/self.trainer.max_epochs)
        # steps_per_epoch = len(train_dataloader)
        pct_start = 0.3
        print("max_lr:", self.max_lr, "epochs:", self.trainer.max_epochs, "steps_per_epoch:", steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                optimizer, 
                                max_lr=self.max_lr,
                                epochs=self.trainer.max_epochs,
                                steps_per_epoch=steps_per_epoch,
                                pct_start=pct_start,
                                div_factor=10,
                                final_div_factor=10,
                                three_phase=False,
                                anneal_strategy='linear'
                            )
        return([optimizer], [{'scheduler': scheduler, 'interval': 'step'}])

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        logits = F.log_softmax(output, dim=1)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = cross_entropy_loss(logits, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.accuracy, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, 'test')

    def evaluate(self, batch, stage):
        x, y = batch
        output = self(x)
        logits = F.log_softmax(output, dim=1)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = cross_entropy_loss(logits, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", self.accuracy, prog_bar=True)
        return logits
    
    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.cifar10_train = CIFAR10(self.data_dir, train=True, transform=self.train_transforms)
            self.cifar10_val = CIFAR10(self.data_dir, train=False, transform=self.test_transforms)
            # self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=BATCH_SIZE, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=BATCH_SIZE, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=BATCH_SIZE, num_workers=os.cpu_count())    
    
class MisclassifiedCollector(Callback):

    def __init__(self):
        super().__init__()
        self.misclassified_data = []
        self.origData = None

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        data, target = batch
        # print("Data shape:", data.shape)

        pred_batch = outputs.argmax(dim=1).cpu().tolist()
        actual_batch = target.cpu().tolist()

        if (len(self.misclassified_data) < 20):
            for i in range(data.shape[0]):
                if pred_batch[i] != actual_batch[i]:
                    _misclassified_data = {
                        'pred': pred_batch[i],
                        'actual': actual_batch[i],
                        'data': data[i].detach().cpu().numpy()
                    }
                    self.misclassified_data.append(_misclassified_data)
        # print("misclassified len:", len(self.misclassified_data))
