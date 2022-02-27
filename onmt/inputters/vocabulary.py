from torchtext.vocab import vocab
from collections import Counter, OrderedDict

from onmt.inputters.text_dataset import SemTextDataset

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
POS_UNK = '<pos0>'


def create_vocab(opt, *datasets):
    """
    Creates a torchtext vocabulary of source and target
    datasets texts. Indices go from most to least frequent word
    :param opt: program dictionary of parameters
    :param datasets: (Dataset) data
    :return: Vocabulary class
    """

    glove_file = "C:/Users/Gio/PycharmProjects/CoMeatIt/glove.6B.50d.txt"

    print("Loading Glove vocabulary")
    glove = {}
    with open(glove_file, encoding="utf8") as f:
        lines = f.readlines()
    for line in lines:
        splits = line.split()
        glove[splits[0]] = 0

    max_size_src = opt.src_vocab_size
    max_size_tgt = opt.tgt_vocab_size
    counter_src = Counter()
    counter_tgt = Counter()
    for dataset in datasets:
        if type(dataset) is SemTextDataset:
            for src_text, target_txt, _, _, _, _, _ in dataset:
                counter_src.update(src_text)
                counter_tgt.update(target_txt)
        else:
            for src_text, target_txt, _, _, _ in dataset:
                counter_src.update(src_text)
                counter_tgt.update(target_txt)
    # remove non english common words
    for token, count in (counter_src & counter_tgt).items():
        if token.lower() not in glove:
            counter_src.pop(token)
            counter_tgt.pop(token)
    sorted_by_freq_words_src = sorted(counter_src.items(), key=lambda x: x[1], reverse=True)
    sorted_by_freq_words_tgt = sorted(counter_tgt.items(), key=lambda x: x[1], reverse=True)
    ordered_dict_words_src = OrderedDict(sorted_by_freq_words_src+[(f"<pos{i}>", 1) for i in range(100)])
    ordered_dict_words_tgt = OrderedDict(sorted_by_freq_words_tgt+[(f"<pos{i}>", 1) for i in range(100)])
    final_vocab_src = vocab(ordered_dict_words_src)
    final_vocab_tgt = vocab(ordered_dict_words_tgt)

    for i, t in enumerate([UNK_WORD, PAD_WORD, EOS_WORD]):
        final_vocab_src.insert_token(t, i)
    for i, t in enumerate([UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD]):
        final_vocab_tgt.insert_token(t, i)

    final_vocab_src.set_default_index(final_vocab_src[UNK_WORD])
    final_vocab_tgt.set_default_index(final_vocab_tgt[UNK_WORD])

    # the previous method consisted in a simple reduction of the tgt using the attention for the generation but rare words could be missing from src_raw and also attention could be badly scattered in src

    # for both src and tgt remove COMMON rare words and add pos1-100
    # in batch substitute in both the missing with incremental pos1-100  
    # at translation time if you predict a pos watch in src_row the pos with maximum attention (aligment)

    return final_vocab_src, final_vocab_tgt


def get_max_index(vocabulary):
    return len(vocabulary.get_itos()) + 1


def get_indices(vocabulary, example):
    """
    Get list of indices corresponding to tokens in vocabulary
    :param vocabulary: Vocabulary class
    :param example: list of tokens
    :return: the list of indices
    """
    """Returns the indices of an example (list of tokens)"""
    return vocabulary.lookup_indices(example)
