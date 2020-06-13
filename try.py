import os.path as osp
import os
from PIL import Image

setname = 'train'
ROOT_PATH = osp.join(os.getcwd(), 'dataset')

csv_path = osp.join( ROOT_PATH, setname + '.csv')
print(csv_path)
lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

data = []
label = []
lb = -1

wnids = []

for l in lines:
    fname, flabel = l.split(',')
    # seg = fname.rsplit(os.sep)
    # print(fname)
    # print(osp.join('.\dataset', fname))

    fpath = os.path.normpath(ROOT_PATH + fname)
    # fpath = osp.join(ROOT_PATH, fname)
    # print(ROOT_PATH)
    if flabel not in wnids:
        wnids.append(flabel)
        lb += 1

    data.append(fpath)
    label.append(lb)

    img = Image.open(fpath).convert('RGB')
    # break