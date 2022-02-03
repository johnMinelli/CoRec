from torchtext.vocab import vocab
from collections import Counter, OrderedDict

from onmt.inputters.text_dataset import SemTextDataset

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


def create_vocab(opt, *datasets):
    """Creates a torchtext vocabulary of source and target
    datasets texts. Indices go from most to least frequent word"""
    max_size_src = opt.src_vocab_size
    max_size_tgt = opt.tgt_vocab_size
    counter_src = Counter()
    counter_tgt = Counter()
    for dataset in datasets:
        if type(dataset) is SemTextDataset:
            for src_text, target_txt, _, _, _, _, _ in dataset:
                counter_src.update([t.lower() for t in src_text])
                counter_tgt.update([t.lower() for t in target_txt])
        else:
            for src_text, target_txt, _, _, _ in dataset:
                counter_src.update([t.lower() for t in src_text])
                counter_tgt.update([t.lower() for t in target_txt])
    sorted_by_freq_words_src = sorted(counter_src.items(), key=lambda x: x[1], reverse=True)
    sorted_by_freq_words_tgt = sorted(counter_tgt.items(), key=lambda x: x[1], reverse=True)
    ordered_dict_words_src = OrderedDict(sorted_by_freq_words_src)
    ordered_dict_words_tgt = OrderedDict(sorted_by_freq_words_tgt)
    final_vocab_src = vocab(ordered_dict_words_src)
    final_vocab_tgt = vocab(ordered_dict_words_tgt)

    for i, t in enumerate([UNK_WORD, PAD_WORD]):
        final_vocab_src.insert_token(t, i)
    for i, t in enumerate([UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD]):
        final_vocab_tgt.insert_token(t, i)

    final_vocab_src.set_default_index(final_vocab_src[UNK_WORD])
    final_vocab_tgt.set_default_index(final_vocab_tgt[UNK_WORD])

    return final_vocab_src, final_vocab_tgt


def get_max_index(vocabulary):
    return len(vocabulary.get_itos()) + 1


def get_indices(vocabulary, example):
    """Returns the indices of an example (list of tokens)"""
    return vocabulary.lookup_indices(example)
