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


@ray.remote(num_gpus=0.25)
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
        self.local_iter_list = [0] * self.num_of_workers  # initialization as 0
        self.min_iter = 0
        self.push_num = [0] * self.worker_per_group  # a list of num of push requests received from worker i, all workers
        self.ps_index = ps_index
        self.pss = []
        self.sum_gradients = []
        sys.stdout = open(f'{self.args.stdout}/ps{self.ps_index:02}_stdout.log', 'a+', 1)
        sys.stderr = open(f'{self.args.stdout}/ps{self.ps_index:02}_stdout.log', 'a+', 1)

    def init_pss(self, ps0, ps1, ps2):
        self.pss = [ps0,ps1, ps2]
        print("init success")

    def apply_gradients(self, *gradients):
        total = 0
        for ele in range(0, len(self.push_num)):
            total = total + self.push_num[ele]
        summed_gradients = [
            torch.stack(gradient_zip).sum(dim=0) / self.args.world_size
            for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.net.set_gradients(summed_gradients)
        self.optimizer.step()
        if total == 0:
            self.sum_gradients = summed_gradients
        if total % 4 ==0 and total!=0:
            self.sum_gradients = [
                torch.stack(gradient_zip).sum(dim=0)
                    for gradient_zip in zip(self.sum_gradients, summed_gradients)
                    ]
            for i in range(3):
                self.pss[i].ps_syn.remote(*self.sum_gradients)
            self.sum_gradients.clear()
        elif total % 4 == 1:
            self.sum_gradients=summed_gradients
        else:
            self.sum_gradients = [
                torch.stack(gradient_zip).sum(dim=0)
                for gradient_zip in zip(self.sum_gradients, summed_gradients)
            ]
        #for i in range(3):
        #    self.pss[i].ps_syn.remote(summed_gradients)

    def ps_syn(self, *gradients):
        self.optimizer.zero_grad()
        self.net.set_gradients(gradients)
        self.optimizer.step()

    def get_weights(self):
        #for name, p in self.net.named_parameters():
        #    print(name, p.size())
        return {k: v.cuda() for k, v in self.net.state_dict().items() if 'weight' in k or 'bias' in k}

    def blocked(self, worker_index, local_iter):
        self.local_iter_list[int(worker_index / 4)] = local_iter
        min_iter = min(self.local_iter_list)
        return local_iter > min_iter + self.bounded_delay



@ray.remote(num_gpus=0.125)
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
        self.writer = SummaryWriter(os.path.join(self.args.log_dir, f'rank_{self.worker_index:02}'))
        self.itr = 0

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
                self.data_iterator = iter(self.train_loader)
                batch = next(self.data_iterator)
                #self.test(self.test_loader)
            src = batch['src'].cuda()
            trg = batch['trg'].cuda()
            self.net.zero_grad()

            outputs = self.net(src, trg)
            outputs = outputs[1:].view(-1, outputs.shape[-1])
            trg = trg[1:].view(-1)
            loss = self.criterion(outputs, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_value)
            train_loss = loss.item()

            pss[self.worker_index%4].apply_gradients.remote(self.net.get_gradients())

            print(
            f'Woker_index:{self.worker_index} Training* loss:{loss.item()} |  train ppl: {math.exp(loss.item())} | time: {time.time()}')
            self.itr += 1

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
    parser.add_argument('--world-size', default=2, type=int,
                        help='node size in simulation')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int, help="train epoch")
    parser.add_argument('--data-dir', default='./data',
                        help='the data directory location')
    parser.add_argument('--log-dir', default='./board/resnet', help='train visual log location')
    parser.add_argument('--checkpoint', default='./checkpoint/resnet', help='checkpoint location')
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

    dirs = [args.data_dir, args.log_dir, args.checkpoint, args.stdout]
    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d, mode=0o755)

    ray.shutdown()
    ray.init(num_gpus=20, ignore_reinit_error=True)
    print('==> ray.init..')

    # get model
    train_data, valid_data, test_data, model, clip_value, PAD_IDX = get_model(args)

    pss = [ParameterServer.remote(args, model, clip_value, PAD_IDX, args.bounded_delay, i) for i in range(args.ps_num)]
    print('==>ps success..')
    pss[0].init_pss.remote(pss[1], pss[2], pss[3])
    pss[1].init_pss.remote(pss[0], pss[2], pss[3])
    pss[2].init_pss.remote(pss[0], pss[1], pss[3])
    pss[3].init_pss.remote(pss[0], pss[1], pss[2])
    worker_tasks = [Worker.remote(i, args, model, clip_value, PAD_IDX)
                    for i in range(args.world_size)]
    print('==>worker_tasks..')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.apply(init_weights)

    for worker in worker_tasks:
        worker.compute_gradients.remote(pss)
    i=0
    while i <= 200:
    #    # Evaluate the current model
        #if i%10 ==0:
        #    current_weights = ray.get(pss[i%ps_num].get_weights.remote())
        #    model.set_weights(current_weights)
        i += 1
        time.sleep(40)

    ray.shutdown()


if __name__ == '__main__':
    main()