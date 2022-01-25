#!/usr/bin/env python

import configargparse
import codecs
import torch
#import onmt.model_builder
#import onmt.translate.beam
from onmt.helpers.model_builder import load_test_model
from itertools import count
import onmt.opts as opts
from onmt.translate.beam import GNMTGlobalScorer
from onmt.helpers.model_builder import load_test_model
from onmt.inputters.text_dataset import TextDataset
from onmt.inputters.input_aux import build_dataset_iter, load_dataset, load_vocab
from onmt.utils.logging import logger


def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    dummy_parser = configargparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    vocab, model, model_opt = load_test_model(opt, dummy_opt.__dict__)

    scorer = GNMTGlobalScorer(opt)

    translator = DiffTranslator(model, opt, model_opt,
                                global_scorer=scorer, out_file=out_file,
                                report_score=report_score, logger=logger)

    return translator


class DiffTranslator(object):

    def __init__(self,
                 model,
                 opt,
                 model_opt,
                 global_scorer=None,
                 out_file=None,
                 report_score=True,
                 logger=None):

        self.opt = opt
        self.model = model

        self.gpu = opt.gpu
        self.cuda = opt.gpu > -1

        self.n_best = opt.n_best
        self.max_length = opt.max_length
        self.max_sent_length = opt.max_sent_length
        self.beam_size = opt.beam_size
        self.min_length = opt.min_length
        self.stepwise_penalty = opt.stepwise_penalty
        self.dump_beam = opt.dump_beam
        self.block_ngram_repeat = opt.block_ngram_repeat
        self.ignore_when_blocking = set(opt.ignore_when_blocking)

        self.replace_unk = opt.replace_unk

        self.verbose = opt.verbose
        self.report_bleu = opt.report_bleu
        self.report_rouge = opt.report_rouge
        self.fast = opt.fast
        # TODO copy attention ?
        # self.copy_attn = model_opt.copy_attn

        self.global_scorer = global_scorer
        self.out_file = out_file
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def semantic(self, test_diff=None, train_diff=None, train_msg=None, preprocessed_data=None, batch_size=None, semantic_msg=None, shard_dir=None):
        """
        save the semantic info
        """
        if preprocessed_data is None and (test_diff is None or train_diff is None or train_msg is None):
            raise AssertionError("--data argument with preprocessed data or" +
                                 "data specific paths [--test_diff, --train_diff, --train_msg] must be specified")

        if batch_size is None:
            raise ValueError("batch_size must be set")

        if preprocessed_data is not None and self.max_sent_length is not None:
            logger.info("--max_sent_length will be ignored since --data parameter specify a dataset already created")

        # load vocab
        vocab = load_vocab(self.opt, None)

        if preprocessed_data is not None:
            data_iter = build_dataset_iter(load_dataset("train", preprocessed_data), vocab, batch_size)
        else:
            # create dataset with messages not truncated
            data_iter = build_dataset_iter(TextDataset(train_diff, train_msg, self.max_sent_length))  # are we able to use the msg here afterwards (would be nice since if we load the pt we have the msg here)? else is better to pass None and use the file later

        # FIXME don't shuffle(?)

        memorys = []
        shard = 0
        # run encoder
        for batch in data_iter:
            src, source_lengths = batch[0]
            enc_states, memory_bank, src_lengths = self.model.encoder(src, src_lengths)

            feature = torch.max(memory_bank, 0)[0]
            _, rank = torch.sort(batch.indices, descending=False)
            feature = feature[rank]
            memorys.append(feature)
            # consider the memory, must shard
            if len(memorys) % 200 == 0:
                # save file
                memorys = torch.cat(memorys)
                torch.save(memorys, shard_dir + "shard.%d" % shard)

                memorys = []
                shard += 1
        if len(memorys) > 0:
            memorys = torch.cat(memorys)
            torch.save(memorys, shard_dir + "shard.%d" % shard)
            shard += 1

        indexes = []
        for i in range(shard):
            print(i)
            shard_index = torch.load(shard_dir + "shard.%d" % i)
            indexes.append(shard_index)
        indexes = torch.cat(indexes)

        # search the best
        if preprocessed_data is not None:
            data_iter = build_dataset_iter(load_dataset("test", preprocessed_data), vocab, batch_size)  # FIXME facciamo il load e lo creiamo in preprocess phase, ok? o vogliamo fare la creazione solo qui?
        else:
            # create dataset with messages not truncated
            data_iter = build_dataset_iter(TextDataset(test_diff, None, self.max_sent_length))

        diffs = []
        msgs = []
        with open(train_msg, 'r') as tm:
            train_msgs = tm.readlines()
        with open(train_diff, 'r') as td:
            train_diffs = td.readlines()

        for batch in data_iter:
            src, source_lengths = batch[0]
            enc_states, memory_bank, src_lengths = self.model.encoder(src, src_lengths)

            feature = torch.max(memory_bank, 0)[0]
            _,  rank = torch.sort(batch.indices, descending=False)
            feature = feature[rank]
            numerator = torch.mm(feature, indexes.transpose(0, 1))
            denominator = torch.mm(feature.norm(2, 1).unsqueeze(1), indexes.norm(2, 1).unsqueeze(1).transpose(0, 1))
            sims = torch.div(numerator, denominator)
            tops = torch.topk(sims, 1, dim=1)
            idx = tops[1][:, -1].tolist()
            # todo get score
            for i in idx:
                diffs.append(train_diffs[i].strip() + '\n')
                msgs.append(train_msgs[i].strip() + '\n')

        with open(semantic_msg, 'w') as sm:
            for i in msgs:
                sm.write(i)
                sm.flush()

        for i in diffs:
            self.out_file.write(i)
            self.out_file.flush()

        return


    def translate(self,
                  src_path=None,
                  src_data_iter=None,
                  tgt_path=None,
                  tgt_data_iter=None,
                  src_dir=None,
                  batch_size=None,
                  attn_debug=False,
                  sem_path=None,
                  src_vocab=None):
        """
        Translate content of `src_data_iter` (if not None) or `src_path`
        and get gold scores if one of `tgt_data_iter` or `tgt_path` is set.

        Note: batch_size must not be None
        Note: one of ('src_path', 'src_data_iter') must not be None

        Args:
            src_path (str): filepath of source data
            src_data_iter (iterator): an interator generating source data
                e.g. it may be a list or an openned file
            tgt_path (str): filepath of target data
            tgt_data_iter (iterator): an interator generating target data
            src_dir (str): source directory path
                (used for Audio and Image datasets)
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        assert src_data_iter is not None or src_path is not None

        if batch_size is None:
            raise ValueError("batch_size must be set")

        test_dataset = TextDataset(src_path, tgt_path)
        vocab = torch.load(src_vocab)  # need to create vocab for test set
        test_loader = build_dataset_iter(test_dataset, vocab, batch_size)

        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"

        # TODO builder
        builder = None
        #
        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        for batch in test_loader:
            batch_data = self.translate_batch(batch, test_dataset, fast=True, attn_debug=False)
            # TODO translations = builder.from_batch(batch_data)
            translations = None
            for trans in translations:
                all_scores += [trans.pred_scores[:self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt_path is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
                all_predictions += [n_best_preds]
                self.out_file.write('\n'.join(n_best_preds) + '\n')
                self.out_file.flush()

        return all_scores, all_predictions


    def translate_batch(self, batch, data, attn_debug, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            if fast:
                # TODO _fast_translate_batch, _translate_batch
                return self._fast_translate_batch(
                    batch,
                    data,
                    self.max_length,
                    min_length=self.min_length,
                    n_best=self.n_best,
                    return_attention=attn_debug or self.replace_unk)
            else:
                return self._translate_batch(batch, data)