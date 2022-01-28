from collections import Counter
import torch, torchtext
from itertools import chain
import codecs
from torch.utils.data import Dataset

# local imports


class TextDataset(Dataset):
    def __init__(self, src_path, target_path, src_max_len=None, target_max_len=None, transform=None, target_transform=None):
        super(TextDataset, self).__init__()

        self.src_path = src_path
        self.target_path = target_path
        self.transform = transform
        self.target_transform = target_transform

        self.src_texts = []
        self.target_texts = []
        with codecs.open(src_path, "r", "utf-8") as cf:
            for i, line in enumerate(cf):
                self.src_texts.append(line.strip().split()[:src_max_len])
        if target_path is not None:
            with codecs.open(target_path, "r", "utf-8") as cf:
                for line in cf:
                    self.target_texts.append(line.strip().split()[:target_max_len])

    def __len__(self):
        assert len(self.src_texts) == len(self.target_texts)
        return len(self.src_texts)

    def __getitem__(self, idx):
        return self.src_texts[idx], self.target_texts[idx], len(self.src_texts[idx]), len(self.target_texts[idx])


class SemTextDataset(TextDataset):
    def __init__(self, src_path, target_path, sem_path, src_max_len=None, target_max_len=None, transform=None, target_transform=None, indexed_data=False):
        super(SemTextDataset, self).__init__(src_path, target_path, src_max_len, target_max_len, transform, target_transform, indexed_data)

        self.sem_path = sem_path
        self.sem_texts = []
        self.indexes = []
        with codecs.open(sem_path, "r", "utf-8") as cf:
            for i, line in enumerate(cf):
                self.sem_texts.append(line.strip().split()[:src_max_len])
                if indexed_data:
                    self.indexes.append(i)

    def __len__(self):
        assert len(self.src_texts) == len(self.target_texts)
        return len(self.src_texts)

    def __getitem__(self, idx):
        return self.src_texts[idx], self.target_texts[idx], self.sem_texts[idx], self.indexes[idx], \
               len(self.src_texts[idx]), len(self.target_texts[idx]), len(self.sem_texts[idx]), len(self.indexes[idx]) \
                   if type(self.target_texts[idx]) == type([]) else 1

