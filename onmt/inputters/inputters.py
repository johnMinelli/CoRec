﻿import glob
import random
import torch, os, codecs, gc, torchtext
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from onmt.utils.logging import logger
from onmt.inputters.vocab import get_indices

def load_vocabulary(vocabulary_path, tag=""):
    """
    Loads a vocabulary from the given path.
    :param vocabulary_path: path to load vocabulary from
    :param tag: tag for vocabulary (only used for logging)
    :return: vocabulary or None if path is null
    """
    vocabulary = None
    if vocabulary_path:
        vocabulary = []
        logger.info("Loading {} vocabulary from {}".format(tag, vocabulary_path))

        if not os.path.exists(vocabulary_path):
            raise RuntimeError("{} vocabulary not found at {}!".format(tag, vocabulary_path))
        else:
            with codecs.open(vocabulary_path, 'r', 'utf-8') as f:
                for line in f:
                    if len(line.strip()) == 0:
                        continue
                    word = line.strip().split()[0]
                    vocabulary.append(word)
    return vocabulary


def lazily_load_dataset(corpus_type, opt):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = opt.data + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def build_dataset_iter(dataset, vocabulary, batch_size):
    # from onmt.inputters.base_dataset import (UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD)

    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    UNK = 0
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'

    def generate_batch(data_batch):
        _, _, en_len, de_len = zip(*data_batch)
        # for padding
        max_en_len = max(en_len)
        max_de_len = max(de_len)
        de_batch, en_batch = [], []
        for (en_item, de_item, en_item_len, de_item_len) in data_batch:
            en_tensor = torch.tensor(get_indices(vocabulary, en_item))
            if en_item_len != max_en_len:

                en_tensor = torch.cat((en_tensor, torch.zeros(max_en_len - en_item_len, dtype=torch.int)))
            de_tensor = torch.tensor(get_indices(vocabulary, de_item))
            if de_item_len != max_de_len:

                de_tensor = torch.cat((de_tensor, torch.zeros(max_de_len - de_item_len, dtype=torch.int)))
            en_batch.append(en_tensor)
            de_batch.append(de_tensor)
            # de_batch.append(torch.cat([torch.tensor([BOS_WORD]), de_item, torch.tensor([EOS_WORD])], dim=0))
            # en_batch.append(torch.cat([torch.tensor([BOS_WORD]), en_item, torch.tensor([EOS_WORD])], dim=0))
        #en_batch = torch.Tensor(en_batch)
        #de_batch = torch.Tensor(de_batch)

        return en_batch, de_batch

    # TODO it need similar length sentences in a batch to have a small padding
#    def batch_sampler():
#        indices = [(i, len(s[1])) for i, s in enumerate(dataset)]
#        random.shuffle(indices)
#        pooled_indices = []
#        # create pool of indices with similar lengths
#        for i in range(0, len(indices), batch_size * 100):
#            pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))

#        pooled_indices = [x[0] for x in pooled_indices]
        # yield indices for current batch
#        for i in range(0, len(pooled_indices), batch_size):
#            yield pooled_indices[i:i + batch_size]  # if you don't yield remove the next(iter( in the trainer

    # return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=generate_batch)
    return DataLoader(dataset, collate_fn=generate_batch, batch_size=batch_size)


def load_vocab(opt, checkpoint):
    if checkpoint is not None:
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
    else:
        vocab = torch.load(opt.data + '.vocab.pt')  # TESTME is this already a torchtext.Vocab type?
    return vocab


