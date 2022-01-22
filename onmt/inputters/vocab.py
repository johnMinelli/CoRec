from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from onmt.inputters.text_dataset import TextDataset


def create_vocab(dataset: TextDataset):
    pass
    #counter_src = Counter()
    #for text in dataset.src_texts:
    #    counter_src.update(text)
