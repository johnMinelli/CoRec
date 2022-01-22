from collections import Counter
import torch, torchtext
from itertools import chain
import codecs
from torch.utils.data import Dataset
# local imports
from onmt.inputters.dataset_base import (DatasetBase, UNK_WORD,
                                         PAD_WORD, BOS_WORD, EOS_WORD)


class MyTextDataset(Dataset):
    def __init__(self, src_path, target_path, src_max_len, target_max_len, transform=None, target_transform=None):

        self.src_path = src_path
        self.target_path = target_path
        self.transform = transform
        self.target_transform = target_transform

        self.src_texts = []
        self.target_texts = []
        with codecs.open(src_path, "r", "utf-8") as cf:
            for line in cf:
                self.src_texts.append(line.strip().split()[:src_max_len])
        with codecs.open(target_path, "r", "utf-8") as cf:
            for line in cf:
                self.target_texts.append(line.strip().split()[:target_max_len])

        super(MyTextDataset, self).__init__()

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        return self.src_texts[idx], self.target_texts[idx]



class TextDataset(DatasetBase):
    """ Dataset for data_type=='text'

        Build `Example` objects, `Field` objects, and filter_pred function
        from text corpus.

        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
                Keys are like 'src', 'tgt', 'src_map', and 'alignment'.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            num_src_feats (int): number of source side features.
            num_tgt_feats (int): number of target side features.
            src_seq_length (int): maximum source sequence length.
            tgt_seq_length (int): maximum target sequence length.
            dynamic_dict (bool): create dynamic dictionaries?
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """

    def __init__(self, fields, src_examples_iter, tgt_examples_iter,
                 num_src_feats=0, num_tgt_feats=0,
                 src_seq_length=0, tgt_seq_length=0,
                 dynamic_dict=False, use_filter_pred=True,
                 syn_examples_iter=None, sem_examples_iter=None,
                 num_syn_feats=0, num_sem_feats=0
                 ):
        self.data_type = 'text'

        # self.src_vocabs: mutated in dynamic_dict, used in
        # collapse_copy_scores and in Translator.py
        self.src_vocabs = []

        self.n_src_feats = num_src_feats
        self.n_tgt_feats = num_tgt_feats
        self.n_syn_feats = num_syn_feats
        self.n_sem_feats = num_sem_feats

        # Each element of an example is a dictionary whose keys represents
        # at minimum the src tokens and their indices and potentially also
        # the src and tgt features and alignment information.
        if syn_examples_iter or sem_examples_iter:
            ext_examples_iter = [syn_examples_iter, sem_examples_iter]
        else:
            ext_examples_iter = None

        if ext_examples_iter is not None:
            if tgt_examples_iter is not None:
                examples_iter = (self._join_dicts(src, ext, tgt) for src, ext, tgt in
                                 zip(src_examples_iter, ext_examples_iter, tgt_examples_iter))
            else:
                examples_iter = (self._join_dicts(list(y for x in item for y in (x if isinstance(x, tuple) else (x,)))) for item in
                                 zip(src_examples_iter, zip(*ext_examples_iter)))
        elif tgt_examples_iter is not None:
            examples_iter = (self._join_dicts(src, tgt) for src, tgt in
                             zip(src_examples_iter, tgt_examples_iter))
        else:
            examples_iter = src_examples_iter

        if dynamic_dict:
            examples_iter = self._dynamic_dict(examples_iter)

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)

        # If out_examples is a generator, we need to save the filter_pred
        # function in serialization too, which would cause a problem when
        # `torch.save()`. Thus we materialize it as a list.
        src_size = 0

        out_examples = []
        for ex_values in example_values:
            example = self._construct_example_fromlist(
                ex_values, out_fields)
            src_size += len(example.src)
            out_examples.append(example)

        def filter_pred(example):
            """ ? """
            return 0 < len(example.src) <= src_seq_length \
                and 0 < len(example.tgt) <= tgt_seq_length

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        super(TextDataset, self).__init__(
            out_examples, out_fields, filter_pred
        )

    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        # Default to a balanced sort, prioritizing tgt len match.
        # TODO: make this configurable.
        if hasattr(ex, "tgt"):
            return len(ex.src), len(ex.tgt)
        return len(ex.src)

    @staticmethod
    def collapse_copy_scores(scores,
                             batch,
                             tgt_vocab,
                             src_vocabs,
                             batch_dim=1,
                             batch_offset=None):
        """
        Given scores from an expanded dictionary
        corresponding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        offset = len(tgt_vocab)
        for b in range(scores.size(batch_dim)):
            blank = []
            fill = []
            batch_id = batch_offset[b] if batch_offset is not None else b
            index = batch.indices.data[batch_id]
            src_vocab = src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    blank.append(offset + i)
                    fill.append(ti)
            if blank:
                blank = torch.Tensor(blank).type_as(batch.indices.data)
                fill = torch.Tensor(fill).type_as(batch.indices.data)
                score = scores[:, b] if batch_dim == 1 else scores[b]
                score.index_add_(1, fill, score.index_select(1, blank))
                score.index_fill_(1, blank, 1e-10)
        return scores

    @staticmethod
    def make_text_examples_nfeats_tpl(text_iter, text_path, truncate, side):
        """
        Args:
            text_iter(iterator): an iterator (or None) that we can loop over
                to read examples.
                It may be an openned file, a string list etc...
            text_path(str): path to file or None
            path (str): location of a src or tgt file.
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Returns:
            (example_dict iterator, num_feats) tuple.
        """
        assert side in ['src', 'tgt', 'syn', 'sem']

        if text_iter is None:
            if text_path is not None:
                text_iter = TextDataset.make_text_iterator_from_file(text_path)
            else:
                return (None, 0)

        # All examples have same number of features, so we peek first one
        # to get the num_feats.
        examples_nfeats_iter = \
            TextDataset.make_examples(text_iter, truncate, side)

        first_ex = next(examples_nfeats_iter)
        num_feats = first_ex[1]

        # Chain back the first element - we only want to peek it.
        examples_nfeats_iter = chain([first_ex], examples_nfeats_iter)
        examples_iter = (ex for ex, nfeats in examples_nfeats_iter)

        return (examples_iter, num_feats)

    @staticmethod
    def make_examples(text_iter, truncate, side):
        """
        Args:
            text_iter (iterator): iterator of text sequences
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Yields:
            (word, features, nfeat) triples for each line.
        """
        for i, line in enumerate(text_iter):
            line = line.strip().split()
            if truncate:
                line = line[:truncate]

            words, feats, n_feats = \
                TextDataset.extract_text_features(line)

            example_dict = {side: words, "indices": i}
            if feats:
                prefix = side + "_feat_"
                example_dict.update((prefix + str(j), f)
                                    for j, f in enumerate(feats))
            yield example_dict, n_feats

    @staticmethod
    def make_text_iterator_from_file(path):
        with codecs.open(path, "r", "utf-8") as corpus_file:
            for line in corpus_file:
                yield line

    @staticmethod
    def get_fields():
        """
        Args:
            n_src_features (int): the number of source features to
                create `torchtext.data.Field` for.
            n_tgt_features (int): the number of target features to
                create `torchtext.data.Field` for.

        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        #fields["src"] = torchtext.data.Field(
        #    pad_token=PAD_WORD,
        #    include_lengths=True)

        #fields["tgt"] = torchtext.data.Field(
        #    init_token=BOS_WORD, eos_token=EOS_WORD,
        #    pad_token=PAD_WORD)

        #fields["indices"] = torchtext.data.Field(
        #    use_vocab=False, dtype=torch.long,
        #    sequential=False)

        return fields

    @staticmethod
    def get_num_features(corpus_file, side):
        """
        Peek one line and get number of features of it.
        (All lines must have same number of features).
        For text corpus, both sides are in text form, thus
        it works the same.

        Args:
            corpus_file (str): file path to get the features.
            side (str): 'src' or 'tgt'.

        Returns:
            number of features on `side`.
        """
        with codecs.open(corpus_file, "r", "utf-8") as cf:

            f_line = cf.readline().strip().split()
            _, _, num_feats = TextDataset.extract_text_features(f_line)

        return num_feats

    # Below are helper functions for intra-class use only.
    def _dynamic_dict(self, examples_iter):
        for example in examples_iter:
            src = example["src"]
            src_vocab = torchtext.vocab.Vocab(Counter(src),
                                              specials=[UNK_WORD, PAD_WORD])
            self.src_vocabs.append(src_vocab)
            # Mapping source tokens to indices in the dynamic dict.
            src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
            example["src_map"] = src_map

            if "tgt" in example:
                tgt = example["tgt"]
                mask = torch.LongTensor(
                    [0] + [src_vocab.stoi[w] for w in tgt] + [0])
                example["alignment"] = mask
            yield example