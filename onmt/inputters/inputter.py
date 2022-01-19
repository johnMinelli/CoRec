import glob
from collections import defaultdict, Counter, OrderedDict
from itertools import count
import torch, os, codecs, gc, torchtext
# local imports
from onmt.inputters.text_dataset import TextDataset
from onmt.inputters.dataset_base import (UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD)
from onmt.utils.logging import logger


def get_fields(data_type, n_src_features, n_tgt_features):
    """
    Args:
        data_type: type of the source input. Options are [text|img|audio].
        n_src_features: the number of source features to
            create `torchtext.data.Field` for.
        n_tgt_features: the number of target features to
            create `torchtext.data.Field` for.

    Returns:
        A dictionary whose keys are strings and whose values are the
        corresponding Field objects.
    """
    if data_type == 'text':
        return TextDataset.get_fields(n_src_features, n_tgt_features)
#    elif data_type == 'img':
#        return ImageDataset.get_fields(n_src_features, n_tgt_features)
#    elif data_type == 'audio':
#        return AudioDataset.get_fields(n_src_features, n_tgt_features)
    else:
        raise ValueError("Data type not implemented")


def load_fields_from_vocab(vocab, data_type="text"):
    """
    Load Field objects from `vocab.pt` file.
    """
    vocab = dict(vocab)
    n_src_features = len(collect_features(vocab, 'src'))
    n_tgt_features = len(collect_features(vocab, 'tgt'))
    fields = get_fields(data_type, n_src_features, n_tgt_features)
    for k, v in vocab.items():
        # Hack. Can't pickle defaultdict :(
        v.stoi = defaultdict(lambda: 0, v.stoi)
        fields[k].vocab = v
    return fields


def save_fields_to_vocab(fields):
    """
    Save Vocab objects in Field objects to `vocab.pt` file.
    """
    vocab = []
    for k, f in fields.items():
        if f is not None and 'vocab' in f.__dict__:
            f.vocab.stoi = f.vocab.stoi
            vocab.append((k, f.vocab))
    return vocab


def collect_features(fields, side="src"):
    """
    Collect features from Field object.
    """
    assert side in ["src", "tgt", "syn", "sem"]
    feats = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feats.append(key)
    return feats


def get_num_features(data_type, corpus_file, side):
    """
    Args:
        data_type (str): type of the source input.
            Options are [text|img|audio].
        corpus_file (str): file path to get the features.
        side (str): for source or for target.

    Returns:
        number of features on `side`.
    """
    assert side in ["src", "tgt", "syn", "sem"]

    if data_type == 'text':
        return TextDataset.get_num_features(corpus_file, side)
#    elif data_type == 'img':
#        return ImageDataset.get_num_features(corpus_file, side)
#    elif data_type == 'audio':
#        return AudioDataset.get_num_features(corpus_file, side)
    else:
        raise ValueError("Data type not implemented")


def _build_field_vocab(field, counter, **kwargs):
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token]
        if tok is not None))
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)


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
        logger.info("Loading {} vocabulary from {}".format(tag,
                                                           vocabulary_path))

        if not os.path.exists(vocabulary_path):
            raise RuntimeError(
                "{} vocabulary not found at {}!".format(tag, vocabulary_path))
        else:
            with codecs.open(vocabulary_path, 'r', 'utf-8') as f:
                for line in f:
                    if len(line.strip()) == 0:
                        continue
                    word = line.strip().split()[0]
                    vocabulary.append(word)
    return vocabulary


def build_vocab(train_dataset_files, fields, data_type, share_vocab,
                src_vocab_path, src_vocab_size, src_words_min_frequency,
                tgt_vocab_path, tgt_vocab_size, tgt_words_min_frequency,
                syn_vocab_path=None, syn_vocab_size=0, syn_words_min_frequency=0,
                sem_vocab_path=None, sem_vocab_size=0, sem_words_min_frequency=0):
    """
    Args:
        train_dataset_files: a list of train dataset pt file.
        fields (dict): fields to build vocab for.
        data_type: "text", "img" or "audio"?
        share_vocab(bool): share source and target vocabulary?
        src_vocab_path(string): Path to src vocabulary file.
        src_vocab_size(int): size of the source vocabulary.
        src_words_min_frequency(int): the minimum frequency needed to
                include a source word in the vocabulary.
        tgt_vocab_path(string): Path to tgt vocabulary file.
        tgt_vocab_size(int): size of the target vocabulary.
        tgt_words_min_frequency(int): the minimum frequency needed to
                include a target word in the vocabulary.

    Returns:
        Dict of Fields
    """
    counter = {}

    # Prop src from field to get lower memory using when training with image
    if data_type == 'img' or data_type == 'audio':
        fields.pop("src")

    for k in fields:
        counter[k] = Counter()

    # Load vocabulary
    src_vocab = load_vocabulary(src_vocab_path, tag="source")
    if src_vocab is not None:
        src_vocab_size = len(src_vocab)
        logger.info('Loaded source vocab has %d tokens.' % src_vocab_size)
        for i, token in enumerate(src_vocab):
            # keep the order of tokens specified in the vocab file by
            # adding them to the counter with decreasing counting values
            counter['src'][token] = src_vocab_size - i

    tgt_vocab = load_vocabulary(tgt_vocab_path, tag="target")
    if tgt_vocab is not None:
        tgt_vocab_size = len(tgt_vocab)
        logger.info('Loaded source vocab has %d tokens.' % tgt_vocab_size)
        for i, token in enumerate(tgt_vocab):
            counter['tgt'][token] = tgt_vocab_size - i

    syn_vocab = load_vocabulary(syn_vocab_path, tag="syn")
    if syn_vocab is not None:
        syn_vocab_size = len(syn_vocab)
        logger.info('Loaded syn vocab has %d tokens.' % syn_vocab_size)
        for i, token in enumerate(syn_vocab):
            counter['syn'][token] = syn_vocab_size - i

    sem_vocab = load_vocabulary(sem_vocab_path, tag="sem")
    if sem_vocab is not None:
        sem_vocab_size = len(sem_vocab)
        logger.info('Loaded sem vocab has %d tokens.' % sem_vocab_size)
        for i, token in enumerate(sem_vocab):
            counter['sem'][token] = sem_vocab_size - i

    for index, path in enumerate(train_dataset_files):
        dataset = torch.load(path)
        logger.info(" * reloading %s." % path)
        for ex in dataset.examples:
            for k in fields:
                val = getattr(ex, k, None)
                if not fields[k].sequential:
                    continue
                elif k == 'src' and src_vocab:
                    continue
                elif k == 'tgt' and tgt_vocab:
                    continue
                counter[k].update(val)

        # Drop the none-using from memory but keep the last
        if (index < len(train_dataset_files) - 1):
            dataset.examples = None
            gc.collect()
            del dataset.examples
            gc.collect()
            del dataset
            gc.collect()

    _build_field_vocab(fields["tgt"], counter["tgt"],
                       max_size=tgt_vocab_size,
                       min_freq=tgt_words_min_frequency)
    logger.info(" * tgt vocab size: %d." % len(fields["tgt"].vocab))

    # All datasets have same num of n_tgt_features,
    # getting the last one is OK.
    for j in range(dataset.n_tgt_feats):
        key = "tgt_feat_" + str(j)
        _build_field_vocab(fields[key], counter[key])
        logger.info(" * %s vocab size: %d." % (key,
                                               len(fields[key].vocab)))

    if data_type == 'text':
        if syn_vocab_size:
            _build_field_vocab(fields["syn"], counter["src"],
                               max_size=syn_vocab_size,
                               min_freq=syn_words_min_frequency)
            logger.info(" * syn vocab size: %d." % len(fields["syn"].vocab))
        if sem_vocab_size:
            _build_field_vocab(fields["sem"], counter["src"],
                               max_size=sem_vocab_size,
                               min_freq=sem_words_min_frequency)
            logger.info(" * sem vocab size: %d." % len(fields["sem"].vocab))

        _build_field_vocab(fields["src"], counter["src"],
                           max_size=src_vocab_size,
                           min_freq=src_words_min_frequency)
        logger.info(" * src vocab size: %d." % len(fields["src"].vocab))

        # All datasets have same num of n_src_features,
        # getting the last one is OK.
        for j in range(dataset.n_src_feats):
            key = "src_feat_" + str(j)
            _build_field_vocab(fields[key], counter[key])
            logger.info(" * %s vocab size: %d." %
                        (key, len(fields[key].vocab)))

        # Merge the input and output vocabularies.
        if share_vocab:
            # `tgt_vocab_size` is ignored when sharing vocabularies
            logger.info(" * merging src and tgt vocab...")
            merged_vocab = merge_vocabs(
                [fields["src"].vocab, fields["tgt"].vocab],
                vocab_size=src_vocab_size,
                min_frequency=src_words_min_frequency)
            fields["src"].vocab = merged_vocab
            fields["tgt"].vocab = merged_vocab

    return fields


def merge_vocabs(vocabs, vocab_size=None, min_frequency=1):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.

    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
        min_frequency: `int` minimum frequency for word to be retained.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = sum([vocab.freqs for vocab in vocabs], Counter())
    return torchtext.vocab.Vocab(merged,
                                 specials=[UNK_WORD, PAD_WORD,
                                           BOS_WORD, EOS_WORD],
                                 max_size=vocab_size,
                                 min_freq=min_frequency)


def build_dataset(fields, data_type, src_data_iter=None, src_path=None,
                  src_dir=None, tgt_data_iter=None, tgt_path=None,
                  src_seq_length=0, tgt_seq_length=0,
                  src_seq_length_trunc=0, tgt_seq_length_trunc=0,
                  dynamic_dict=False, sample_rate=0,
                  window_size=0, window_stride=0, window=None,
                  normalize_audio=True, use_filter_pred=True,
                  image_channel_size=3,
                  syn_path=None, sem_path=None,
                  syn_data_iter=None, sem_data_iter=None):
    """
    Build src/tgt examples iterator from corpus files, also extract
    number of features.
    """

    def _make_examples_nfeats_tpl(data_type, src_data_iter, src_path, src_dir,
                                  src_seq_length_trunc, sample_rate,
                                  window_size, window_stride,
                                  window, normalize_audio,
                                  image_channel_size=3):
        """
        Process the corpus into (example_dict iterator, num_feats) tuple
        on source side for different 'data_type'.
        """

        if data_type == 'text':
            src_examples_iter, num_src_feats = \
                TextDataset.make_text_examples_nfeats_tpl(
                    src_data_iter, src_path, src_seq_length_trunc, "src")

        # elif data_type == 'img':
        #    src_examples_iter, num_src_feats = \
        #        ImageDataset.make_image_examples_nfeats_tpl(
        #            src_data_iter, src_path, src_dir, image_channel_size)

        # elif data_type == 'audio':
        #    if src_data_iter:
        #        raise ValueError("""Data iterator for AudioDataset isn't
        #                            implemented""")

        #    if src_path is None:
        #        raise ValueError("AudioDataset requires a non None path")
        #    src_examples_iter, num_src_feats = \
        #        AudioDataset.make_audio_examples_nfeats_tpl(
        #            src_path, src_dir, sample_rate,
        #            window_size, window_stride, window,
        #            normalize_audio)

        return src_examples_iter, num_src_feats

    src_examples_iter, num_src_feats = \
        _make_examples_nfeats_tpl(data_type, src_data_iter, src_path, src_dir,
                                  src_seq_length_trunc, sample_rate,
                                  window_size, window_stride,
                                  window, normalize_audio,
                                  image_channel_size=image_channel_size)

    # For all data types, the tgt side corpus is in form of text.
    tgt_examples_iter, num_tgt_feats = \
        TextDataset.make_text_examples_nfeats_tpl(
            tgt_data_iter, tgt_path, tgt_seq_length_trunc, "tgt")
    if syn_path:
        syn_examples_iter, num_syn_feats = \
            TextDataset.make_text_examples_nfeats_tpl(
                syn_data_iter, syn_path, src_seq_length_trunc, "syn")
    else:
        syn_examples_iter, num_syn_feats = None, None
    if sem_path:
        sem_examples_iter, num_sem_feats = \
            TextDataset.make_text_examples_nfeats_tpl(
                sem_data_iter, sem_path, src_seq_length_trunc, "sem")
    else:
        sem_examples_iter, num_sem_feats = None, None

    if data_type == 'text':
        dataset = TextDataset(fields, src_examples_iter, tgt_examples_iter,
                              num_src_feats, num_tgt_feats,
                              src_seq_length=src_seq_length,
                              tgt_seq_length=tgt_seq_length,
                              dynamic_dict=dynamic_dict,
                              use_filter_pred=use_filter_pred,
                              syn_examples_iter=syn_examples_iter, sem_examples_iter=sem_examples_iter,
                              num_syn_feats=num_syn_feats, num_sem_feats=num_sem_feats
                              )

    #elif data_type == 'img':
    #    dataset = ImageDataset(fields, src_examples_iter, tgt_examples_iter,
    #                           num_src_feats, num_tgt_feats,
    #                           tgt_seq_length=tgt_seq_length,
    #                           use_filter_pred=use_filter_pred,
    #                           image_channel_size=image_channel_size)

    #elif data_type == 'audio':
    #    dataset = AudioDataset(fields, src_examples_iter, tgt_examples_iter,
    #                           tgt_seq_length=tgt_seq_length,
    #                           use_filter_pred=use_filter_pred)

    return dataset


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


def _load_fields(dataset, opt, checkpoint):
    if checkpoint is not None:
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = load_fields_from_vocab(
            checkpoint['vocab'])
    else:
        fields = load_fields_from_vocab(
            torch.load(opt.data + '.vocab.pt'))
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in dataset.examples[0].__dict__])

    return fields
