import argparse
import torch
from torch.utils.data import DataLoader
import os
from mydataset import MyDataset
from samplers import TestSampler
from protonet import Protonet
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric, Timer
from PIL import Image
import samplers
import realtime
import cv2
import numpy as np
import random

seed = 80   # 80
torch.manual_seed(seed) # cpu
torch.cuda.manual_seed(seed) #gpu
np.random.seed(seed) #numpy
random.seed(seed)
torch.backends.cudnn.deterministic=True # cudnn

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model_acc = []
    way = 20
    for i in range(1, 6):
        shot = i
        query = 1
        load_path = "./save/proto-" + str(shot) + "-200/max-acc.pth"

        dataset = MyDataset('test', './data20_v2/')
        sampler = TestSampler(dataset.label, 100, way, shot + query)
        loader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=8,
                            pin_memory=True)

        model = Protonet().cuda()
        model.load_state_dict(torch.load(load_path))
        model.eval()

        k = way * shot     
        ave_acc = Averager()

        for i, batch in enumerate(loader, 1):
            data, _ = [_.cuda() for _ in batch]
            data_shot, data_query = data[:k], data[k:]
            
            t = Timer()

            x = model(data_shot)
            x = x.reshape(shot, way, -1).mean(dim=0)
            p = x

            logits = euclidean_metric(model(data_query), p)

            pred = torch.argmax(logits, dim=1)

            label = torch.arange(way) # 0~11 because l[shot] == l[query]
            label = label.type(torch.cuda.LongTensor)

            acc = count_acc(logits, label)
            ave_acc.add(acc)
            x = None; p = None; logits = None
        
        model_acc.append(ave_acc.item() * 100)


    for i in range(1, 6):
        print('{}-way {}-shot learning, average accuracy: {}'.format(
            way, i, model_acc[i-1]
        ))
