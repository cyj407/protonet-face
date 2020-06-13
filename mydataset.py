import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import ntpath
import os


ROOT_PATH = './dataset/'


class MyDataset(Dataset):

    def __init__(self, setname):
        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        wnids = []

        for l in lines:
            fname, flabel = l.split(',')
            # seg = fname.rsplit('/')
            # fname = osp.join(seg[1], seg[2])
            path = os.path.normpath(ROOT_PATH + fname)
            # path = osp.join(os.getcwd(), 'dataset', fname)
            if flabel not in wnids:
                wnids.append(flabel)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label