import argparse
import torch
from torch.utils.data import DataLoader
import os
from mydataset import MyDataset
from samplers import MySampler
from protonet import Protonet
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric, Timer
import samplers
import realtime
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import random

seed = 80   # 80
torch.manual_seed(seed) # cpu
torch.cuda.manual_seed(seed) #gpu
np.random.seed(seed) #numpy
random.seed(seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn


def getDataset(each_class_num):
    csv_path = os.path.join('data20_me', 'test.csv')
    lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
    data = []       # path of the image
    person = samplers.doc_order
    photo = samplers.photo_num

    marked = [False] * len(lines)
    for i in range(len(person)):
        idx = (person[i]-1) * each_class_num + photo[i] - 1
        marked[idx] = True
        l = lines[idx]
        fname, flabel = l.split(',')
        path = os.path.normpath('./res' + fname)
        data.append(path)
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu', default='1')
    parser.add_argument('--load', default='./save/proto-me-500/max-acc.pth')
    # parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--way', type=int, default=21)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=1)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    ####　preprocess
    dataset = MyDataset('test', './data20_me/')
    sampler = MySampler(dataset.label, 1, args.way, args.shot + args.query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=8,
                        pin_memory=True)

    model = Protonet().cuda()
    model.load_state_dict(torch.load(args.load))
    model.eval()

    data, l = [_.cuda() for _ in next(iter(loader))]
    k = args.way * args.shot
    data_shot = data[:k]

    x = model(data_shot)
    x = x.reshape(args.shot, args.way, -1).mean(dim=0)
    p = x

    # open camera
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('frame')

    while(cv2.getWindowProperty('frame', 0) >= 0):
        
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        target = None
        target = realtime.real_time_detect_haar(frame)
        # target = realtime.real_time_detect_dnn(frame)

        if(target != None):
            data_query = data[k:-1]
            data_query = torch.cat([data_query, target])
            logits = euclidean_metric(model(data_query), p) # [20, 20]

            max_dis_idx = torch.argmax(logits, dim=1)[-1]
            max_dis = logits[-1][max_dis_idx]
            sum_dis = logits[-1].sum()
            other_avr_dis = (sum_dis - logits[-1][max_dis_idx]) / (args.way -1)
            # print('max distance: {}, other distance average: {}'.format(
                # max_dis, other_avr_dis))
            # if(max_dis - other_avr_dis < 20):
                # text = 'Unknown'
            # else:
            text = samplers.ground_truth[max_dis_idx]
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()