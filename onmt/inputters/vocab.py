from torchtext.vocab import vocab
from collections import Counter, OrderedDict


def create_vocab(datasets):
    """Creates a torchtext vocabulary of source and target
    datasets texts. Indices go from most to least frequent word"""

    counter = Counter()
    for dataset in datasets:

        for src_text, target_txt in dataset:
            counter.update(src_text)
            counter.update(target_txt)

    sorted_by_freq_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict_words = OrderedDict(sorted_by_freq_words)

    return vocab(ordered_dict_words)


def get_indices(vocabulary, example):
    """Returns the indices of an example (list of tokens)"""
    return vocabulary.lookup_indices(example)
