import ray

import torch.optim as optim
# import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torchvision.transforms as transforms
from torchvision import models as image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, Batch

import spacy
import os
import argparse
import sys
from filelock import FileLock
import time
import types
import math
from functools import partial
from models.seq2seq import *

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

@ray.remote(num_gpus=0.1)
class GlobalPS(object):
    def __init__(self, model, args, num_ps):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=model.to(device)
        self.args=args
        self.num_ps=num_ps
        self.optimizer = optim.Adam(self.model.parameters())

    def aggregate(self,*gradients):
        self.optimizer.zero_grad()
        self.model.set_gradients(gradients)
        self.optimizer.step()
        return self.model.get_weights()

@ray.remote(num_gpus=0.1)
class ParameterServer(object):
    def __init__(self, args, model, clip_value, PAD_IDX, stale_low_bound, ps_index):
        self.args = args
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = model.to(device)
        self.train_loader, self.valid_loader, self.test_loader = get_datasets(args, 0)
        self.clip_value = clip_value
        self.PAD_IDX = PAD_IDX
        self.optimizer = optim.Adam(self.net.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.num_of_workers = args.world_size
        self.num_of_groups = 4
        self.worker_per_group = 4
        self.stale_low_bound = stale_low_bound  # low_bound, that is, sl
        self.local_workers = [ps_index, self.num_of_groups + ps_index, self.num_of_groups * 2 + ps_index,
                              self.num_of_groups * 2 + ps_index]
        self.min_iter = 0
        self.r_oversl = [0] * self.worker_per_group  # the num of extra iterations worker p is allowed beyond sL
        self.push_num = [0] * self.worker_per_group  # a list of num of push requests received from worker i, all workers
        self.timestamp_list = [[0 for i in range(2)] for i in range(self.worker_per_group)]  # the timestamps of two latest push requests by all workers, 2*num_of_workers
        self.ps_index = ps_index
        self.globalps=[]
        self.sum_gradients = []
        sys.stdout = open(f'{self.args.stdout}/ps{self.ps_index:02}_stdout.log', 'a+', 1)
        sys.stderr = open(f'{self.args.stdout}/ps{self.ps_index:02}_stdout.log', 'a+', 1)
        self.epoch_timelist = [[-1] * self.num_of_groups for i in
                               range(self.worker_per_group)]  # timelist of all workers
        self.epoch_grouplist = [-1] * self.num_of_groups  # mean iteration time of the group


    def init_pss(self, globalps):
        self.globalps = globalps
        print("init success")

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
            self.net.set_weights(ray.get(self.globalps.aggregate.remote(*(self.sum_gradients))))
            self.sum_gradients.clear()
        elif total % 4 == 1:
            self.sum_gradients = summed_gradients
        else:
            self.sum_gradients = [
                torch.stack(gradient_zip).sum(dim=0)
                for gradient_zip in zip(self.sum_gradients, summed_gradients)
            ]

    def ps_syn(self, *gradients):
        self.optimizer.zero_grad()
        self.net.set_gradients(gradients)
        self.optimizer.step()

    def get_weights(self):
        return {k: v.cuda() for k, v in self.net.state_dict().items() if 'weight' in k or 'bias' in k}

    def desparsify(self, values, indices, ctx):
        #values, indices = tensors
        numel, shape = ctx
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices.long(), values)
        #print("after shape,",indices.long().dtype())
        return tensor_decompressed.view(shape)


    def get_time(self, worker_index, epoch_time):#decide the fastest or the slowest
        self.epoch_timelist[worker_index%4][int(worker_index/4)] = epoch_time
        workercount=0
        sumtime=0
        for i in range(4):
            #some workers haven't finish 1 epoch
            if self.epoch_timelist[i].count(-1) !=0:
                return -1
        for i in range(4):
            #if self.epoch_timelist[worker_index%4][i] == -1:
            #    print("false")
            #    continue
            #else:
            workercount +=1
            sumtime += self.epoch_timelist[worker_index%4][i]
        self.epoch_grouplist[worker_index%4] = sumtime/workercount
        #print("self.epoch_timelist", self.epoch_timelist)
        #print("self.epoch_grouplist", self.epoch_grouplist)
        if  self.epoch_grouplist[worker_index%4] == max(self.epoch_grouplist):
            return 1
        elif self.epoch_grouplist[worker_index%4] == min(self.epoch_grouplist):
            return 0
        else:
            return 2

    def blocked(self, worker_index, local_iter):
        self.local_iter_list[int(worker_index / 4)] = local_iter
        min_iter = min(self.local_iter_list)
        return local_iter > min_iter + self.bounded_delay


@ray.remote(num_gpus=0.08)
class Worker(object):
    def __init__(self, worker_index, args, model, clip_value, PAD_IDX):
        self.args = args
        self.worker_index = worker_index
        self.train_loader, self.valid_loader, self.test_loader = get_datasets(args, worker_index)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = model.to(device)
        self.learning_rate = args.lr
        self.clip_value = clip_value
        self.PAD_IDX = PAD_IDX
        self.data_iterator = iter(self.train_loader)
        self.optimizer = optim.Adam(self.net.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        sys.stdout = open(f'{self.args.stdout}/{self.worker_index:02}_stdout.log', 'a+', 1)
        sys.stderr = open(f'{self.args.stdout}/{self.worker_index:02}_stdout.log', 'a+', 1)
        self.itr = 0
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
            elif self.epoch_count ==4:
                self.compress_ratio = 0.015625
            elif self.epoch_count >=5:
                self.compress_ratio = 0.004
        elif self.slowest:
            self.sample_ratio = 0.01
            if self.epoch_count == 2:
                self.compress_ratio = 0.35
            elif self.epoch_count == 3:
                self.compress_ratio = 0.1225
            elif self.epoch_count ==4:
                self.compress_ratio = 0.042875
            elif self.epoch_count >=5:
                self.compress_ratio = 0.015
        else:
            self.sample_ratio = 0.009
            if self.epoch_count == 2:
                self.compress_ratio = 0.3
            elif self.epoch_count == 3:
                self.compress_ratio = 0.09
            elif self.epoch_count ==4:
                self.compress_ratio = 0.027
            elif self.epoch_count >=5:
                self.compress_ratio = 0.0081

    def compute_gradients(self, pss):
        while True:
            self.net.train()
            if ray.get(pss[self.worker_index%4].blocked.remote(self.worker_index, self.itr)):
                #print("worker blocked",self.worker_index, self.worker_iter)
                time.sleep(0.05)
                continue
            weights = ray.get(pss[self.worker_index % 4].get_weights.remote())
            self.net.set_weights(weights)
            try:
                batch = next(self.data_iterator)
            except StopIteration:  # When the epoch ends, start a new epoch.
                pss[self.worker_index % 4].apply_gradients.remote(self.worker_index, time.time(),
                                                                  self.net.get_gradients())
                # push all gradients at the end of a epoch
                self.net.zero_grad()
                self.epoch_count += 1
                if self.epoch_time == -1:
                    self.epoch_time = time.time() - self.epoch_start
                else:
                    self.epoch_time = self.epoch_time * (1 - self.time_momentum) + (
                                                                                   time.time() - self.epoch_start) * self.time_momentum
                self.train_loader, self.valid_loader, self.test_loader = get_datasets(self.args,(self.worker_index+self.epoch_count)%self.args.world_size)
                self.data_iterator = iter(self.train_loader)
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
                # self.test(self.test_loader)
                # print("epoch_time,", self.epoch_time)
                # self.dynamic_ratio = ray.get(pss[0].get_time.remote(self.worker_index, self.epoch_time))
                # print("epoch_grouplist", self.epoch_grouplist)
                self.epoch_start = time.time()
                batch = next(self.data_iterator)
            src = batch['src'].cuda()
            trg = batch['trg'].cuda()
            #self.net.zero_grad()

            outputs = self.net(src, trg)
            outputs = outputs[1:].view(-1, outputs.shape[-1])
            trg = trg[1:].view(-1)
            loss = self.criterion(outputs, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_value)
            train_loss = loss.item()
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
                indices = newtensor[1]
                p.grad.view(-1).index_fill_(0, indices.long(), 0)  # local
                tensors.append(newtensor)
            pss[self.worker_index%4].apply_gradients.remote(tensors)
            print(
            f'Woker_index:{self.worker_index} Training* loss:{loss.item()} |  train ppl: {math.exp(loss.item())} | time: {time.time()} | compress: {self.compress_ratio} | sample: {self.sample_ratio} | itr:{self.itr}')

            self.itr += 1

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
                #print("num_indices",num_indices,"num_selects",num_selects)
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
        indices = indices.float()
        ctx = numel, shape
        return values, indices, ctx

    def test(self, test_loader):
        self.net.eval()
        epoch_loss = 0
        loader_len = len(test_loader)
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                src = batch['src'].cuda()
                trg = batch['trg'].cuda()

                output = self.net(src, trg, 0)  # turn off teacher forcing

                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)

                loss = self.criterion(output, trg)
                epoch_loss += loss.item()
            print(
            f'Testing* loss:{epoch_loss/loader_len} | ppl: {math.exp(epoch_loss/loader_len)} | time: {time.time()} | itr: {self.itr}')

        return epoch_loss / loader_len


def get_model(args):
    enc_emb_dim = args.enc_emb
    dec_emb_dim = args.dec_emb
    enc_hid_dim = args.enc_hid
    dec_hid_dim = args.dec_hid
    enc_dropout = args.enc_drop
    dec_dropout = args.dec_drop
    clip_value = 1

    # torch.backends.cudnn.deterministic = True
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    # Tokenize text from a string into a list of strings in lambda
    source = Field(tokenize=lambda text: [tok.text for tok in spacy_de.tokenizer(text)],
                   init_token='<sos>',
                   eos_token='<eos>',
                   lower=True)

    target = Field(tokenize=lambda text: [tok.text for tok in spacy_en.tokenizer(text)],
                   init_token='<sos>',
                   eos_token='<eos>',
                   lower=True)

    train_data, valid_data, test_data = Multi30k.splits(root='./data',
                                                        exts=('.de', '.en'),
                                                        fields=(source, target))

    source.build_vocab(train_data, min_freq=2)
    target.build_vocab(train_data, min_freq=2)
    input_dim = len(source.vocab)
    output_dim = len(target.vocab)

    attn = Attention(enc_hid_dim, dec_hid_dim)
    enc = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout)
    dec = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(enc, dec, device)
    model = ray_wrapper(model)

    PAD_IDX = target.vocab.stoi['<pad>']

    return train_data, valid_data, test_data, model, clip_value, PAD_IDX


def get_datasets(args, rank):
    batch_size = args.batch_size

    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    # Tokenize text from a string into a list of strings in lambda
    source = Field(tokenize=lambda text: [tok.text for tok in spacy_de.tokenizer(text)],
                   init_token='<sos>',
                   eos_token='<eos>',
                   lower=True)

    target = Field(tokenize=lambda text: [tok.text for tok in spacy_en.tokenizer(text)],
                   init_token='<sos>',
                   eos_token='<eos>',
                   lower=True)

    train_data, valid_data, test_data = Multi30k.splits(root='./data',
                                                        exts=('.de', '.en'),
                                                        fields=(source, target))
    source.build_vocab(train_data, min_freq=2)
    target.build_vocab(train_data, min_freq=2)

    def torchtext_collate(data):
        b = Batch(data, train_data)
        return {'src': b.src, 'trg': b.trg}

    sampler = DistributedSampler(train_data, num_replicas=args.world_size, rank=rank)
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=torchtext_collate,
                              sampler=sampler, shuffle=False,
                              num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, collate_fn=torchtext_collate,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=torchtext_collate,
                             shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, valid_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='GSSP for nlp')
    parser.add_argument('--model', default='resnet', help='model name')
    parser.add_argument('--world-size', default=16, type=int,
                        help='node size in simulation')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int, help="train epoch")
    parser.add_argument('--data-dir', default='./data',
                        help='the data directory location')
    parser.add_argument('--stdout', default='./stdout/resnet', help='stdout log dir for subprocess')
    parser.add_argument('--momentum', default=0.9, type=float, help='the momentum of iteration time')
    parser.add_argument('--enc-emb', default=256, type=int, help='Encoder embedding size')
    parser.add_argument('--dec-emb', default=256, type=int, help='Decoder embedding size')
    parser.add_argument('--enc-hid', default=512, type=int, help='Encoder hidden layer size')
    parser.add_argument('--dec-hid', default=512, type=int, help='Decoder hidden layer size')
    parser.add_argument('--enc-drop', default=0.5, type=float, help='Encoder dropout probability ')
    parser.add_argument('--dec-drop', default=0.5, type=float, help='Decoder dropout probability ')
    parser.add_argument('--ps-num', default=4, type=int)
    parser.add_argument('--bounded-delay', default=3, type=int)

    args = parser.parse_args()
    sys.stdout = open(f'{args.stdout}/main_stdout.log', 'a+', 1)
    sys.stderr = open(f'{args.stdout}/main_stdout.log', 'a+', 1)

    dirs = [args.data_dir, args.stdout]
    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d, mode=0o755)

    ray.shutdown()
    ray.init(num_gpus=20, ignore_reinit_error=True)
    print('==> ray.init..')

    # get model
    train_data, valid_data, test_data, model, clip_value, PAD_IDX = get_model(args)
    worker_tasks = [Worker.remote(i, args, model, clip_value, PAD_IDX)
                    for i in range(args.world_size)]
    globalps=GlobalPS.remote(model, args, args.ps_num)
    pss = [ParameterServer.remote(args, model, clip_value, PAD_IDX, args.bounded_delay, i) for i in range(args.ps_num)]
    print('==>ps success..')
    pss[0].init_pss.remote(globalps)
    pss[1].init_pss.remote(globalps)
    pss[2].init_pss.remote(globalps)
    pss[3].init_pss.remote(globalps)


    print('==>worker_tasks..')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.apply(init_weights)

    for worker in worker_tasks:
        worker.compute_gradients.remote(pss)
    i=0
    while i <= 1000:
        i += 1
        time.sleep(40)

    ray.shutdown()


if __name__ == '__main__':
    main()


