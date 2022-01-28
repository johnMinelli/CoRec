from torchtext.vocab import vocab
from collections import Counter, OrderedDict

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'


def create_vocab(*datasets):
    """Creates a torchtext vocabulary of source and target
    datasets texts. Indices go from most to least frequent word"""

    counter = Counter()
    for dataset in datasets:

        for src_text, target_txt, _, _ in dataset:
            counter.update(src_text)
            counter.update(target_txt)
    counter.update([PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD])
    sorted_by_freq_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict_words = OrderedDict(sorted_by_freq_words)

    final_vocab = vocab(ordered_dict_words)
    final_vocab.set_default_index(get_max_index(final_vocab))
    return final_vocab

def create_sem_vocab(*datasets):
    """Creates a torchtext vocabulary of source and target
    datasets texts. Indices go from most to least frequent word"""

    counter = Counter()
    for dataset in datasets:

        for src_text, target_txt, sem_txt, _, _, _, _ in dataset:
            counter.update(src_text)
            counter.update(target_txt)
            counter.update(sem_txt)
    counter.update([PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD])
    sorted_by_freq_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict_words = OrderedDict(sorted_by_freq_words)

    final_vocab = vocab(ordered_dict_words)
    final_vocab.set_default_index(get_max_index(final_vocab))
    return final_vocab


def get_max_index(vocabulary):
    return len(vocabulary.get_itos()) + 1


def get_indices(vocabulary, example):
    """Returns the indices of an example (list of tokens)"""
    return vocabulary.lookup_indices(example)
