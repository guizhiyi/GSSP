import ray
import math
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

import torch.nn.functional as F
from models.vgg import *
from models.resnet import *
from models.lenet import *



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
        self.sched = MultiStepLR(self.optimizer, [70], gamma=0.1)  # TODO
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
        self.epoch_timelist = [[-1] * self.num_of_groups for i in
                               range(self.worker_per_group)]  # timelist of all workers
        self.epoch_grouplist = [-1] * self.num_of_groups  # mean iteration time of the group
        self.initial_ratio = 0.4
        self.comp_ratio = [0.4, 0.4, 0.4, 0, 4]

    def init_pss(self, ps0, ps1, ps2):
        self.pss = [ps0, ps1, ps2]

    def desparsify(self, values, indices, ctx):
        # values, indices = tensors
        numel, shape = ctx
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices.long(), values)
        # print("after shape,",indices.long().dtype())
        return tensor_decompressed.view(shape)

    def apply_gradients(self, *gradients):
        total = 0
        for ele in range(0, len(self.push_num)):
            total = total + self.push_num[ele]
        degradients = []
        for tensors in gradients:
            for values, indices, ctx in tensors:
                afterdesparsify = self.desparsify(values, indices, ctx)
                degradients.append(afterdesparsify)
        summed_gradients = [
            torch.stack(gradient_zip).sum(dim=0) / self.args.world_size
            for gradient_zip in zip(degradients)
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
                self.pss[i].ps_syn.remote(*(self.sum_gradients))
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

    def apply_allgradients(self, *gradients):
        summed_gradients = [
            torch.stack(gradient_zip).sum(dim=0) / self.args.world_size
            for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.net.set_gradients(summed_gradients)
        self.optimizer.step()
        for i in range(3):
            self.pss[i].ps_allsyn.remote(*gradients)

    def ps_allsyn(self, *gradients):
        summed_gradients = [
            torch.stack(gradient_zip).sum(dim=0) / self.args.world_size
            for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.net.set_gradients(summed_gradients)
        self.optimizer.step()
        # return self.net.get_weights()

    def ps_syn(self, summed_gradients):
        self.optimizer.zero_grad()
        self.net.set_gradients(summed_gradients)
        self.optimizer.step()

    def get_weights(self):
        return {k: v.cuda() for k, v in self.net.state_dict().items() if 'weight' in k or 'bias' in k}

    def lr_sched(self):
        self.sched.step()
        print(self.sched.get_last_lr()[0])

    def blocked(self, worker_index, local_iter):
        self.local_iter_list[int(worker_index / 4)] = local_iter
        min_iter = min(self.local_iter_list)
        return local_iter > min_iter + self.bounded_delay

    def get_time(self, worker_index, epoch_time):  # decide the fastest or the slowest
        self.epoch_timelist[worker_index % 4][int(worker_index / 4)] = epoch_time
        workercount = 0
        sumtime = 0
        for i in range(4):
            # some workers haven't finish 1 epoch
            if self.epoch_timelist[i].count(-1) != 0:
                return -1
        for i in range(4):
            workercount += 1
            sumtime += self.epoch_timelist[worker_index % 4][i]
        self.epoch_grouplist[worker_index % 4] = sumtime / workercount

        if self.epoch_grouplist[worker_index % 4] == max(self.epoch_grouplist):
            return 1
        elif self.epoch_grouplist[worker_index % 4] == min(self.epoch_grouplist):
            return 0
        else:
            return 2


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
        self.compress_ratio = 0.35
        self.sample_ratio = 0.01
        self.compress_upper_bound = 1.3
        self.compress_lower_bound = 0.8
        self.max_adaptation_iters = 10
        self.resample = True
        self.attributes = {}
        self.time_momentum = 0.7
        self.epoch_count = 0
        self.epoch_start = time.time()
        self.epoch_time = -1
        self.epoch_grouplist = [-1, -1, -1, -1]
        self.fastest = False
        self.slowest = False
        self.worker_speed = -1

    def warmup(self):
        if self.fastest:
            self.sample_ratio = 0.008
            if self.epoch_count == 2:
                self.compress_ratio = 0.25
            elif self.epoch_count == 3:
                self.compress_ratio = 0.0625
            elif self.epoch_count == 4:
                self.compress_ratio = 0.015625
            elif self.epoch_count >= 5:
                self.compress_ratio = 0.004
        elif self.slowest:
            self.sample_ratio = 0.01
            if self.epoch_count == 2:
                self.compress_ratio = 0.35
            elif self.epoch_count == 3:
                self.compress_ratio = 0.1225
            elif self.epoch_count == 4:
                self.compress_ratio = 0.042875
            elif self.epoch_count >= 5:
                self.compress_ratio = 0.015
        else:
            self.sample_ratio = 0.009
            if self.epoch_count == 2:
                self.compress_ratio = 0.3
            elif self.epoch_count == 3:
                self.compress_ratio = 0.09
            elif self.epoch_count == 4:
                self.compress_ratio = 0.027
            elif self.epoch_count >= 5:
                self.compress_ratio = 0.0081

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
                pss[self.worker_index % 4].apply_allgradients.remote(self.net.get_gradients())
                self.net.zero_grad()
                self.epoch_count += 1
                if self.epoch_time == -1:
                    self.epoch_time = time.time() - self.epoch_start
                else:
                    self.epoch_time = self.epoch_time * (1 - self.time_momentum) + (
                                                                                   time.time() - self.epoch_start) * self.time_momentum

                test_loader, train_loader = get_dataset(self.args, (self.worker_index + self.epoch_count) % self.args.world_size)
                self.data_iterator = iter(train_loader)
                self.worker_speed = ray.get(pss[0].get_time.remote(self.worker_index, self.epoch_time))
                if self.worker_speed == -1:
                    pass
                else:
                    if self.worker_speed == 0:
                        self.fastest = True
                        self.slowest = False
                    elif self.worker_speed == 1:
                        self.fastest = False
                        self.slowest = True
                    else:
                        self.fastest = False
                        self.slowest = False
                    self.warmup()
                self.epoch_start = time.time()
                #self.test(test_loader)
                inputs, targets = next(self.data_iterator)
            inputs, targets = inputs.cuda(), targets.cuda()
            # self.net.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            train_loss = loss.item()
            _, predicted = outputs.max(1)
            total = targets.size(0)
            correct = predicted.eq(targets).sum().item()
            tensors = []
            # count = 0
            for name, p in self.net.named_parameters():
                tensor = None if p.grad is None else p.grad.data
                # print(self.worker_index,"tensor,",tensor)
                numel = tensor.numel()
                shape = list(tensor.size())
                if self.sample_ratio < 1.0:
                    pct_numel = int(math.ceil(numel * self.sample_ratio))
                    cpr_numel = int(math.ceil(2 / self.compress_ratio))
                    if numel <= cpr_numel:
                        num_samples = numel
                    else:
                        num_samples = pct_numel
                else:
                    num_samples = numel
                top_k_samples = int(math.ceil(num_samples * self.compress_ratio))
                num_selects = int(math.ceil(numel * self.compress_ratio))
                self.attributes[name] = (numel, shape, num_selects, num_samples, top_k_samples)
                newtensor = self.sparsify(tensor, name)
                values = newtensor[0]
                indices = newtensor[1]
                p.grad.view(-1).index_fill_(0, indices.long(), 0)  # local
                tensors.append(newtensor)
            pss[self.worker_index % 4].apply_gradients.remote(tensors)
            print(
                f'Woker_index:{self.worker_index} Training* loss:{loss.item()} | acc: {correct / total} | time: {time.time()} |itr: {self.itr} | epoch: {self.epoch_count} | sample ratio: {self.sample_ratio} | compress_ratio: {self.compress_ratio}')
            self.itr = self.itr + 1

    def sparsify(self, tensor, name):
        tensor = tensor.view(-1)
        numel, shape, num_selects, num_samples, top_k_samples = self.attributes[name]

        importance = tensor.abs()
        if numel == num_samples:
            samples = importance
        else:
            samples = importance[torch.randint(0, numel, (num_samples,), device=tensor.device)]

        threshold = torch.min(torch.topk(samples, top_k_samples, 0, largest=True, sorted=False)[0])
        mask = torch.ge(importance, threshold)
        indices = mask.nonzero().view(-1)
        num_indices = indices.numel()

        if numel > num_samples:
            for _ in range(self.max_adaptation_iters):
                # print("num_indices",num_indices,"num_selects",num_selects)
                if num_indices > num_selects:
                    if num_indices > num_selects * self.compress_upper_bound:
                        if self.resample:
                            indices = indices[
                                torch.topk(importance[indices], num_selects,
                                           0, largest=True, sorted=False)[1]
                            ]
                            break
                        else:
                            threshold = threshold * self.compress_upper_bound
                    else:
                        break
                elif num_indices < self.compress_lower_bound * num_selects:
                    threshold = threshold * self.compress_lower_bound
                else:
                    break
                mask = torch.ge(importance, threshold)
                indices = mask.nonzero().view(-1)
                num_indices = indices.numel()

        indices = indices[:num_selects]
        values = tensor[indices]
        # print("name,",name,"indices",indices,"value,",values,"indices.dtype", indices.dtype)
        indices = indices.float()
        # print(indices.dtype)
        # print(indices.dtype)
        # tensor = values, indices
        ctx = numel, shape

        return values, indices, ctx

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


def main():
    parser = argparse.ArgumentParser(description='GSSP for CV')
    parser.add_argument('--model', default='resnet', help='model name')
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

    model = get_model()
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