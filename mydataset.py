from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import ntpath
import os
import cv2

class MyDataset(Dataset):

    def __init__(self, setname, root_path):
        csv_path = os.path.join(root_path, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1
        label_name = []

        wnids = []

        for l in lines:
            fname, flabel = l.split(',')
            path = os.path.normpath(root_path + fname)
            if flabel not in wnids:
                wnids.append(flabel)
                lb += 1
                label_name.append(flabel)
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.label_name = label_name

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