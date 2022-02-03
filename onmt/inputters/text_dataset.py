from collections import Counter
import torch, torchtext
from itertools import chain
import codecs
from torch.utils.data import Dataset

# local imports


class TextDataset(Dataset):
    def __init__(self, src_path, target_path=None, src_max_len=None, target_max_len=None, transform=None, target_transform=None):
        super(TextDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.indexes = []
        self.sort_index = 3

        self.src_path = src_path
        self.target_path = target_path

        self.src_texts = []
        self.target_texts = []
        with codecs.open(src_path, "r", "utf-8") as cf:
            for i, line in enumerate(cf):
                self.src_texts.append([w.lower() for w in line.strip().split()[:src_max_len]])
                self.indexes.append(i)
        if target_path is not None:
            with codecs.open(target_path, "r", "utf-8") as cf:
                for line in cf:
                    self.target_texts.append([w.lower() for w in line.strip().split()[:target_max_len]])

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        return self.src_texts[idx], self.target_texts[idx] if self.target_path is not None else None, self.indexes[idx], \
               len(self.src_texts[idx]), len(self.target_texts[idx]) if self.target_path is not None else 1

class SemTextDataset(TextDataset):
    def __init__(self, src_path, target_path=None, sem_path=None, src_max_len=None, target_max_len=None, transform=None, target_transform=None):
        super(SemTextDataset, self).__init__(src_path, target_path, src_max_len, target_max_len, transform, target_transform)
        self.sort_index = 4

        self.sem_path = sem_path
        self.sem_texts = []
        if sem_path is not None:
            with codecs.open(sem_path, "r", "utf-8") as cf:
                for i, line in enumerate(cf):
                    self.sem_texts.append([w.lower() for w in line.strip().split()])

    def __getitem__(self, idx):
        return self.src_texts[idx],\
               self.target_texts[idx] if self.target_path is not None else None,\
               self.sem_texts[idx] if self.sem_path is not None else None,\
               self.indexes[idx], \
               len(self.src_texts[idx]), \
               len(self.target_texts[idx]) if self.target_path is not None else 1, \
               len(self.sem_texts[idx]) if self.sem_path is not None else 1

