import torch, os, codecs
from torch.utils.data import DataLoader, Sampler
from torchtext.data.utils import RandomShuffler
from onmt.inputters.vocabulary import BOS_WORD, EOS_WORD, PAD_WORD
from onmt.inputters.text_dataset import SemTextDataset
from onmt.utils.logging import logger
from onmt.inputters.vocabulary import get_indices

def load_vocab(vocab_file, checkpoint=None):
    if checkpoint is not None:
        logger.info('Loading vocab from checkpoint')
        vocab = checkpoint['vocab']
    else:
        vocab = torch.load(vocab_file)
    return vocab

def load_dataset(corpus_type, opt):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    pt = opt.data + '.' + corpus_type + '.pt'
    dataset = torch.load(pt)
    logger.info('Loading %s dataset from %s, number of examples: %d' % (corpus_type, pt, len(dataset)))
    return dataset


class MinPaddingSampler(Sampler):

    def __init__(self, data_source, batch_size, shuffle_batches):

        super().__init__(data_source)
        self.dataset = data_source
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches

    def __iter__(self):
        indices = [(i, s[self.dataset.sort_index]) for i, s in enumerate(self.dataset)]
        # sort dataset indices by decreasing length, so that
        # batches contain texts with close lengths, and low padding
        if self.shuffle_batches:
            sorted_indices = sorted(indices, key=lambda x: x[1], reverse=True)
            # select only the actual indexes, not lengths
            sorted_indices = [x[0] for x in sorted_indices]
            for i in RandomShuffler()(range(0, len(indices), self.batch_size)):
                yield sorted_indices[i:i + self.batch_size]
        else:
            for i in range(0, len(indices), self.batch_size):
                sorted_indices = sorted(indices[i:i + self.batch_size], key=lambda x: x[1], reverse=True)
                # sorted_indices = sorted(indices, key=lambda x: x[1], reverse=False)[i:i + self.batch_size]
                yield [x[0] for x in sorted_indices]

    def __len__(self):
        # each time batch size elements are sampled
        return self.batch_size


def build_dataset_iter(dataset, vocabs, batch_size, gpu=False, shuffle_batches=True):
    device = torch.device("cuda" if gpu else "cpu")
    def generate_batch(data_batch):
        _, _, _, src_len, tgt_len = zip(*data_batch)
        # for padding
        max_src_len = max(src_len)
        max_tgt_len = max(tgt_len)
        src_batch, tgt_batch, indexes = [], [], []
        for (src_item, tgt_item, index, src_item_len, tgt_item_len) in data_batch:

            indexes.append(index)
            # encode source
            src_tensor = torch.tensor(get_indices(vocabs["src"], src_item+[EOS_WORD]))
            if src_item_len != max_src_len:
                # creates an array of "blank" tokens inside an array (we need padding[0] to get the actual padding)
                padding = torch.full((1, max_src_len - src_item_len), vocabs["src"][PAD_WORD], dtype=torch.int)
                src_tensor = torch.cat((src_tensor, padding[0]))

            # encode target
            if tgt_item is None:
                tgt_tensor = torch.tensor([])
            else:
                tgt_tensor = torch.tensor(get_indices(vocabs["tgt"], [BOS_WORD]+tgt_item+[EOS_WORD]))
                if tgt_item_len != max_tgt_len:
                    # creates an array of "blank" tokens inside an array (we need padding[0] to get the actual padding)
                    padding = torch.full((1, max_tgt_len - tgt_item_len), vocabs["tgt"][PAD_WORD], dtype=torch.int)[0]
                    tgt_tensor = torch.cat((tgt_tensor, padding))
            src_batch.append(src_tensor)
            tgt_batch.append(tgt_tensor)
        src_batch = torch.cat([tensor.unsqueeze(1) for tensor in src_batch], 1).unsqueeze(2)
        tgt_batch = torch.cat([tensor.unsqueeze(1) for tensor in tgt_batch], 1).unsqueeze(2)
        return {"src_batch": src_batch.to(device), "src_len": torch.tensor(src_len).to(device), "tgt_batch": tgt_batch.to(device),
                "tgt_len": torch.tensor(tgt_len).to(device), "indexes": torch.tensor(indexes).to(device)}

    def generate_batch_sem_dataset(data_batch):
        _, _, _, _, src_len, tgt_len, sem_len = zip(*data_batch)
        # for padding
        max_src_len = max(src_len)
        max_tgt_len = max(tgt_len)
        max_sem_len = max(sem_len)
        src_batch, tgt_batch, sem_batch, indexes = [], [], [], []
        for (src_item, tgt_item, sem_item, index, src_item_len, tgt_item_len, sem_item_len) in data_batch:

            indexes.append(index)
            # encode source
            src_tensor = torch.tensor(get_indices(vocabs["src"], src_item+[EOS_WORD]))
            if src_item_len != max_src_len:
                # creates an array of "blank" tokens inside an array (we need padding[0] to get the actual padding)
                padding = torch.full((1, max_src_len - src_item_len), vocabs["src"][PAD_WORD], dtype=torch.int)[0]
                src_tensor = torch.cat((src_tensor, padding))
            if tgt_item is None:
                tgt_tensor = torch.tensor([])
            else:
                tgt_tensor = torch.tensor(get_indices(vocabs["tgt"], [BOS_WORD]+tgt_item+[EOS_WORD]))
                if tgt_item_len != max_tgt_len:
                    # creates an array of "blank" tokens inside an array (we need padding[0] to get the actual padding)
                    padding = torch.full((1, max_tgt_len - tgt_item_len), vocabs["tgt"][PAD_WORD], dtype=torch.int)[0]
                    tgt_tensor = torch.cat((tgt_tensor, padding))
            if sem_item is None:
                sem_tensor = torch.tensor([])
            else:
                sem_tensor = torch.tensor(get_indices(vocabs["src"], sem_item+[EOS_WORD]))
                if sem_item_len != max_sem_len:
                    # creates an array of "blank" tokens inside an array (we need padding[0] to get the actual padding)
                    padding = torch.full((1, max_sem_len - sem_item_len), vocabs["src"][PAD_WORD], dtype=torch.int)[0]
                    sem_tensor = torch.cat((sem_tensor, padding))
            src_batch.append(src_tensor)
            tgt_batch.append(tgt_tensor)
            sem_batch.append(sem_tensor)
        src_batch = torch.cat([tensor.unsqueeze(1) for tensor in src_batch], 1).unsqueeze(2)
        tgt_batch = torch.cat([tensor.unsqueeze(1) for tensor in tgt_batch], 1).unsqueeze(2)
        sem_batch = torch.cat([tensor.unsqueeze(1) for tensor in sem_batch], 1).unsqueeze(2)
        return {"src_batch": src_batch.to(device), "src_len": torch.tensor(src_len).to(device), "tgt_batch": tgt_batch.to(device), "tgt_len": torch.tensor(tgt_len).to(device),
                "sem_batch": sem_batch.to(device), "sem_len": torch.tensor(sem_len).to(device), "indexes": torch.tensor(indexes)}

    sampler = MinPaddingSampler(dataset, batch_size, shuffle_batches)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=generate_batch_sem_dataset if type(dataset) is SemTextDataset else generate_batch)


