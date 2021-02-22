import ray

import torch
import torch.nn as nn
import torch.optim as optim
# import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
import torchvision.transforms as transforms
from torchvision import models as image
from tensorboardX import SummaryWriter

import os
import argparse
import sys
from filelock import FileLock
import time
import types
from models.vgg import *
from models.resnet import *
from models.lenet import *
import torch.nn.functional as F


def ray_wrapper(obj):
    """Module wapper for Ray."""

    def get_weights(self):
        return {k: v.cuda() for k, v in self.state_dict().items() if 'weight' in k or 'bias' in k}

    def set_weights(self, weights):
        self.load_state_dict(weights, strict=False)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.named_parameters()):
            if g is not None:
                p[1].grad = g

    obj.get_weights = types.MethodType(get_weights, obj)
    obj.set_weights = types.MethodType(set_weights, obj)
    obj.get_gradients = types.MethodType(get_gradients, obj)
    obj.set_gradients = types.MethodType(set_gradients, obj)
    return obj


@ray.remote
class Logger(object):
    def __init__(self):
        pass


@ray.remote(num_gpus=1)
class ParameterServer(object):
    def __init__(self, i, args, model, bounded_delay):
        self.ps_index = i
        self.net = model.cuda()
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        if args.dataset=="cifar10" and args.model=="resnet18":
            self.sched = MultiStepLR(self.optimizer, [50, 70], gamma=0.1)  # TODO
        if args.dataset=="cifar10" and args.model=="vgg19":
            self.sched = MultiStepLR(self.optimizer, [70,100], gamma=0.1)  # TODO
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        self.num_of_workers = args.world_size
        self.num_of_groups = 4
        self.worker_per_group = 4
        self.push_num = [0] * self.worker_per_group
        self.pss = []
        self.bounded_delay = bounded_delay
        self.local_iter_list = self.worker_per_group * [0]
        self.sum_gradients = []
        sys.stdout = open(f'{self.args.stdout}/ps{self.ps_index:02}_stdout.log', 'a+', 1)
        sys.stderr = open(f'{self.args.stdout}/ps{self.ps_index:02}_stdout.log', 'a+', 1)


    def init_pss(self, ps0, ps1, ps2):
        self.pss = [ps0, ps1, ps2]

    def apply_gradients(self, worker_index, *gradients):
        total = 0
        for ele in range(0, len(self.push_num)):
            total = total + self.push_num[ele]
        itr = self.push_num[int((worker_index - self.ps_index) / 4)]
        summed_gradients = [
            torch.stack(gradient_zip).sum(dim=0) / self.args.world_size
            for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.net.set_gradients(summed_gradients)
        self.optimizer.step()
        if total == 0:
            self.sum_gradients = summed_gradients
        if total % 4 == 0 and total != 0:
            self.sum_gradients = [
                torch.stack(gradient_zip).sum(dim=0)
                for gradient_zip in zip(self.sum_gradients, summed_gradients)
            ]
            for i in range(3):
                self.pss[i].ps_syn.remote(i, worker_index, itr, time.time(), *(self.sum_gradients))
            self.sum_gradients.clear()
        elif total % 4 == 1:
            # self.sum_gradients.append(gradients)
            self.sum_gradients = summed_gradients
            # print("sumgradi",summed_gradients)
        else:
            self.sum_gradients = [
                torch.stack(gradient_zip).sum(dim=0)
                for gradient_zip in zip(self.sum_gradients, summed_gradients)
            ]

            # return self.net.get_weights()

    def ps_syn(self, i, worker_index, summed_gradients):
        self.optimizer.zero_grad()
        self.net.set_gradients(summed_gradients)
        self.optimizer.step()

    def lr_sched(self):
        self.sched.step()
        print("ps_index,", self.ps_index, self.sched.get_last_lr()[0])

    def get_weights(self):
        return {k: v.cuda() for k, v in self.net.state_dict().items() if 'weight' in k or 'bias' in k}

    def blocked(self, worker_index, local_iter):
        self.local_iter_list[int(worker_index / 4)] = local_iter
        min_iter = min(self.local_iter_list)
        return local_iter > min_iter + self.bounded_delay


@ray.remote(num_gpus=1)
class Worker(object):
    def __init__(self, worker_index, args, model):
        self.worker_index = worker_index
        self.test_loader, self.trainloader = get_dataset(args, worker_index)
        self.net = model.cuda()
        self.learning_rate = args.lr
        self.criterion = nn.CrossEntropyLoss()
        self.data_iterator = iter(self.trainloader)
        self.args = args
        sys.stdout = open(f'{self.args.stdout}/{self.worker_index:02}_stdout.log', 'a+', 1)
        sys.stderr = open(f'{self.args.stdout}/{self.worker_index:02}_stdout.log', 'a+', 1)
        self.itr = 0
        self.epochs = 0

    def compute_gradients(self, pss):
        while True:
            self.net.train()
            if ray.get(pss[self.worker_index % 4].blocked.remote(self.worker_index, self.itr)):
                # print("worker blocked",self.worker_index, self.worker_iter)
                continue
            weights = ray.get(pss[self.worker_index % 4].get_weights.remote())
            self.net.set_weights(weights)
            try:
                inputs, targets = next(self.data_iterator)
            except StopIteration:  # When the epoch ends, start a new epoch.
                test_loader, train_loader = get_dataset(self.args, self.worker_index)
                self.data_iterator = iter(train_loader)
                self.epochs = self.epochs + 1
                if self.worker_index == 0 and self.args.dataset=="cifar10":
                    pss[0].lr_sched.remote()
                    pss[1].lr_sched.remote()
                    pss[2].lr_sched.remote()
                    pss[3].lr_sched.remote()
                #self.test(test_loader)
                inputs, targets = next(self.data_iterator)
            inputs, targets = inputs.cuda(), targets.cuda()
            self.net.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            train_loss = loss.item()
            _, predicted = outputs.max(1)
            total = targets.size(0)
            correct = predicted.eq(targets).sum().item()
            pss[self.worker_index % 4].apply_gradients.remote(self.worker_index, self.net.get_gradients())
            print(
                f'Woker_index:{self.worker_index} Training* loss:{loss.item()} | acc: {correct / total} | time: {time.time()} |itr: {self.itr} | epoch: {self.epochs}')
            self.itr = self.itr + 1

    def test(self, test_loader):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        # dataloader, trainloader = get_dataset(self.args, 0)
        loader_len = len(test_loader)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                # print(self.net.get_weights()['layer4.0.bn2.num_batches_tracked'])
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                inner_total = targets.size(0)
                inner_correct = predicted.eq(targets).sum().item()
                total += inner_total
                correct += inner_correct
        print('Loss: %.4f | Acc: %.4f%% (%d/%d) | Time: %f' %
              (test_loss / loader_len, correct / total, correct, total, time.time()), self.itr / self.args.world_size)


def get_model(args):
    if args.model== 'lenet':
        net = LeNet()
    elif args.model == 'resnet':
        net = ResNet18()
    elif args.model == 'vgg':
        net = vgg19_bn()
    net = ray_wrapper(net)
    return net


# dataloader
def get_dataset(args, rank):
    if args.dataset=="mnist":
        print('==>get_dataset..')
        mnist_transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transforms)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transforms)

        with FileLock(os.path.expanduser("~/data.lock")):
            sampler = torch.utils.data.DistributedSampler(trainset, num_replicas=args.world_size, rank=rank)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False,
                                                      num_workers=1, sampler=sampler)
            testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
        print('==>testloader, trainloader..')
        return testloader, trainloader
    elif args.dataset=="cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        with FileLock(os.path.expanduser("~/data.lock")):
            sampler = torch.utils.data.DistributedSampler(trainset, num_replicas=args.world_size, rank=rank)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False,
                                                      num_workers=2, sampler=sampler)
            testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
        print('==>testloader, trainloader..')
        return testloader, trainloader


def main():
    parser = argparse.ArgumentParser(description='GSSP for CV')
    parser.add_argument('--model', default='lenet', help='model name')
    parser.add_argument('--world-size', default=16, type=int,
                        help='node size in simulation')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int, help="train epoch")
    parser.add_argument('--download', default=False, action='store_true',
                        help="only download dataset")
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--data-dir', default='./data',
                        help='the data directory location')
    parser.add_argument('--stdout', default='./stdout/resnet', help='stdout log dir for subprocess')
    parser.add_argument('--momentum', default=0.9, type=float, help='the momentum of iteration time')
    args = parser.parse_args()
    sys.stdout = open(f'{args.stdout}/main_stdout.log', 'a+', 1)
    sys.stderr = open(f'{args.stdout}/main_stdout.log', 'a+', 1)
    parser.add_argument('--ps-num', default=4, type=int)
    parser.add_argument('--bounded-delay', default=3, type=int)

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    dirs = [args.data_dir, args.stdout]
    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d, mode=0o755)
    model = get_model(args)
    # ray.init(redis_address="10.108.137.72:6379",ignore_reinit_error=True)
    ray.init(num_gpus=20, ignore_reinit_error=True)
    print('==> ray.init..')
    pss = [ParameterServer.remote(i, args, model, args.bounded_delay) for i in range(args.ps_num)]
    pss[0].init_pss.remote(pss[1], pss[2], pss[3])
    pss[1].init_pss.remote(pss[0], pss[2], pss[3])
    pss[2].init_pss.remote(pss[0], pss[1], pss[3])
    pss[3].init_pss.remote(pss[0], pss[1], pss[2])
    print('==>ps success..')
    worker_tasks = [Worker.remote(i, args, model)
                    for i in range(args.world_size)]
    print('==>worker_tasks..')
    time.sleep(10)

    # model = model.cuda()
    for worker in worker_tasks:
        worker.compute_gradients.remote(pss)
    i = 0
    while i <= 10000:
        i += 1
        time.sleep(30)
    ray.shutdown()


if __name__ == '__main__':
    main()