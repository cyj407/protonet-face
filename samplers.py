import torch
import numpy as np
import os

ground_truth = []
gt_path = []

doc_order = []
photo_num = []

def setPath(name_list, p_name, number):
    global doc_order, photo_num
    p_num = int(name_list.index(p_name)) + 1
    doc_order.append(p_num)

    query_num = int(number.item()) + 1
    photo_num.append(query_num)


class TestSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.name = [ "Justin Bieber", "Ariana Grande",
                    "Ed Sheeran", "Billie Eilish",
                    "Shawn Mendes", "Camilia Cabello",
                    "Troye Sivan", "Dua Lipa",
                    "Charlie Puth", "Selena Gomez",
                    "Zayn Malik", "Jennifer Lawrence",
                    "Donald Trump", "Dakota Johnson",
                    "Niall Horan", "Rihanna",
                    "Gigi Hadid", "Jimmy Fallon",
                    "Ann Hathaway", "Bradley Cooper" ]
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            global ground_truth
            for c in classes:
                ground_truth.append(self.name[c])
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                setPath(self.name, ground_truth[-1], pos[-1])
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class TrainSampler():
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class MySampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.name = [ "Justin Bieber", "Ariana Grande",
                    "Ed Sheeran", "Billie Eilish",
                    "Shawn Mendes", "Camilia Cabello",
                    "Troye Sivan", "Dua Lipa",
                    "Charlie Puth", "Selena Gomez",
                    "Zayn Malik", "Jennifer Lawrence",
                    "Donald Trump", "Dakota Johnson",
                    "Niall Horan", "Rihanna",
                    "Gigi Hadid", "Jimmy Fallon",
                    "Ann Hathaway", "Bradley Cooper", "Chen Yu Jie" ]

        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            global ground_truth
            for c in classes:
                ground_truth.append(self.name[c])
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                setPath(self.name, ground_truth[-1], pos[-1])
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch