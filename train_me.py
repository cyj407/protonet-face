import argparse
import os.path as osp
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from torch.utils.data import Dataset
from mydataset import MyDataset
from samplers import TrainSampler
from protonet import Protonet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=50)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=1)
    parser.add_argument('--way', type=int, default=21)
    parser.add_argument('--save-path', default='./save/proto-me-200')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    ensure_path(args.save_path)

    trainset = MyDataset('train', './data20_me/')
    train_sampler = TrainSampler(trainset.label, 25,
                                      args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              pin_memory=True)

    valset = MyDataset('val','./data20_me/')
    val_sampler = TrainSampler(valset.label, 50,
                                    args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            pin_memory=True)

    model = Protonet().cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    cut = args.shot * args.way
    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            data_shot, data_query = data[:cut], data[cut:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.way, -1).mean(dim=0)

            logits = euclidean_metric(model(data_query), proto)

            label = torch.arange(args.way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            proto = None; logits = None; loss = None

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            data_shot, data_query = data[:cut], data[cut:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.way, -1).mean(dim=0)

            logits = euclidean_metric(model(data_query), proto)
            
            label = torch.arange(args.way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)
            
            proto = None; logits = None; loss = None

        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))