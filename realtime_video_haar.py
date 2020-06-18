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

seed = 10   # 80
torch.manual_seed(seed) # cpu
torch.cuda.manual_seed(seed) #gpu
np.random.seed(seed) #numpy
random.seed(seed)
torch.backends.cudnn.deterministic=True # cudnn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', default='./save/proto-5-1000/max-acc.pth')
    parser.add_argument('--way', type=int, default=20)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=1)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    ####　preprocess
    dataset = MyDataset('test', './data20_v2/')
    sampler = TestSampler(dataset.label, 1, args.way, args.shot + args.query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=8, pin_memory=True)

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
    path = 'videos/billie2.mp4'
    cap = cv2.VideoCapture(path)
    cv2.namedWindow('demo_video')

    while(cap.isOpened()):
        
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 360))


        target = None
        target = realtime.real_time_detect_haar(frame)

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
            if(max_dis - other_avr_dis > 20):
            # if(not isUnknown):
                text = samplers.ground_truth[max_dis_idx]
            else:
                text = 'Unknown'
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        cv2.imshow('demo_video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()