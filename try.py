import os.path as osp

from PIL import Image

setname = 'train'
ROOT_PATH = './dataset/'

csv_path = osp.join(ROOT_PATH, setname + '.csv')
lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
print(lines[-1])

data = []
label = []
lb = -1

wnids = []

for l in lines:
    fname, flabel = l.split(',')
    seg = fname.rsplit('/')
    fname = osp.join(seg[1], seg[2])
    path = osp.join(ROOT_PATH, 'images', fname)
    if flabel not in wnids:
        wnids.append(flabel)
        lb += 1
    data.append(path)
    label.append(lb)

img = Image.open(path).convert('RGB')