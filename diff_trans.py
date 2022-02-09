#!/usr/bin/env python
import copy
import math
import os.path

import configargparse
import numpy as np
import torch
from statistics import mean

from onmt.helpers.report_manager import build_report_manager
from onmt.utils.logging import logger
from itertools import count
import onmt.opts as opts
from onmt.inputters import vocabulary
from onmt.helpers.model_builder import load_test_model
from onmt.inputters.text_dataset import SemTextDataset, TextDataset
from onmt.inputters.input_aux import build_dataset_iter, load_dataset, load_vocab
from onmt.encoders.transformer import TransformerEncoder
from onmt.utils.misc import tile
from onmt.translate.translation_wrapper import TranslationBuilder
from onmt.hashes.smooth import compute_bleu_score


def build_translator(opt, report_score=True):
    dummy_parser = configargparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    model, model_details = load_test_model(opt, dummy_opt.__dict__)

    translator = DiffTranslator(model, opt, model_details, report_score=report_score)

    return translator


class DiffTranslator(object):

    def __init__(self,
                 model,
                 opt,
                 model_details,
                 report_score=True):

        self.shard_dir_default = "data/sem_shard/"
        self.sem_diff_default = "sem.diff"
        self.sem_msg_default = "sem.msg"
        self.test_dataset = TextDataset(opt.src, opt.tgt, opt.max_sent_length)
        self.src_vocab = load_vocab(opt.src_vocab)

        if opt.sem_path is not None:
            if not os.path.isdir(opt.sem_path):
                os.mkdir(opt.sem_path)
            files = os.listdir(opt.sem_path)
            if len(files) > 0:
                assert self.sem_msg_default in files and self.sem_diff_default in files, \
                    "Empty the sem_path folder specified to recompute the data or check that all semantic files are present in the folder"

        self.opt = opt
        self.model = model
        self.gpu = opt.gpu
        self.verbose = opt.verbose
        model_opts, model_train_stats = model_details
        self.copy_attn = model_opts.copy_attn

        self.report_score = report_score
        self.report_manager = build_report_manager(opt, "translate")
        self.report_manager.report_model_details(model_stats=model_train_stats, semantic=opt.sem_path is not None)

        if not opt.semantic_only and opt.sem_path is not None:
            self.lam_sem = self.opt.lam_sem
            self.sem_decoder = copy.deepcopy(self.model.decoder)


    def offline_semantic_retrieval(self, test_diff=None, train_diff=None, train_msg=None, batch_size=None,
                                   semantic_out_dir=None):
        """
        Saves the semantic info in three files: diffs and msgs of training set samples aligned with the test set samples
        by similarity of encode and the shared vocabulary used for translation
           sem.msg
           sem.diff
           shared_sem_vocab.pt
        """
        if test_diff is None or train_diff is None or train_msg is None:
            raise AssertionError("data files paths [--test_diff, --train_diff, --train_msg] must be specified")

        if batch_size is None:
            raise ValueError("batch_size must be set")

        max_sent_length = self.opt.max_sent_length

        # load/create dataset and create iterator
        ds = TextDataset(train_diff, src_max_len=max_sent_length)

        data_iter = build_dataset_iter(ds, self.src_vocab, batch_size, gpu=self.gpu, shuffle_batches=False)

        memories = []
        shard = 0
        if not os.path.exists(self.shard_dir_default):
            os.makedirs(self.shard_dir_default)
        # run encoder
        for batch in data_iter:
            src = batch["src_batch"]
            source_lengths = batch["src_len"]
            batch_indices = batch["indexes"]
            enc_states, memory_bank, src_lengths = self.model.encoder(src, source_lengths)

            feature = torch.max(memory_bank, 0)[0]
            _, rank = torch.sort(batch_indices, descending=False)
            feature = feature[rank]
            memories.append(feature)
            # consider the memory, must shard
            if len(memories) % 200 == 0:  # save file
                memories = torch.cat(memories)
                torch.save(memories, self.shard_dir_default + "shard.%d" % shard)
                print(f"Saving shard {self.shard_dir_default}shard.{shard}")
                memories = []
                shard += 1
        if len(memories) > 0:
            memories = torch.cat(memories)
            torch.save(memories, self.shard_dir_default + "shard.%d" % shard)
            print(f"Saving shard {self.shard_dir_default}shard.{shard}")
            shard += 1

        train_encodings_indexes = []
        for i in range(shard):
            shard_index = torch.load(self.shard_dir_default + "shard.%d" % i)
            train_encodings_indexes.append(shard_index)
        train_encodings_indexes = torch.cat(train_encodings_indexes)

        # get ordered train diffs and msgs from source in order to make use of full length data
        with open(train_msg, 'r') as tm:
            train_msgs = tm.readlines()
        with open(train_diff, 'r') as td:
            train_diffs = td.readlines()

        # search the best (most similar) correspondence of test set encodings with computed training set encodings
        data_iter = build_dataset_iter(self.test_dataset, self.src_vocab, batch_size, gpu=self.gpu, shuffle_batches=False)

        diffs = []
        msgs = []
        for batch in data_iter:
            src = batch["src_batch"]
            source_lengths = batch["src_len"]
            batch_indices = batch["indexes"]
            enc_states, memory_bank, src_lengths = self.model.encoder(src, source_lengths)
            # get the token with maximum attention for all samples in batch
            feature = torch.max(memory_bank, 0)[0]
            # reorder attention results as the order of samples in dataset
            _, rank = torch.sort(batch_indices, descending=False)
            feature = feature[rank]
            # compute similarities
            numerator = torch.mm(feature, train_encodings_indexes.transpose(0, 1))
            denominator = torch.mm(feature.norm(2, 1).unsqueeze(1),
                                   train_encodings_indexes.norm(2, 1).unsqueeze(1).transpose(0, 1))
            sims = torch.div(numerator, denominator)
            # get indices of most similar
            tops = torch.topk(sims, 1, dim=1)
            idx = tops[1][:, -1].tolist()
            for i in idx:
                diffs.append(train_diffs[i].strip() + '\n')
                msgs.append(train_msgs[i].strip() + '\n')

        with open(os.path.join(semantic_out_dir, self.sem_msg_default), 'w') as sm:
            for i in msgs:
                sm.write(i)
                sm.flush()

        with open(os.path.join(semantic_out_dir, self.sem_diff_default), 'w') as of:
            for i in diffs:
                of.write(i)
                of.flush()

        return

    def translate(self, test_diff=None, test_msg=None, sem_path=None, batch_size=None, attn_debug=False, out_file=None):
        """
        Translate content of `src_data_iter` (if not None) or `test_diff`
        and get gold scores if one of `tgt_data_iter` or `test_msg` is set.

        Note: batch_size must not be None
        Note: one of ('test_diff', 'src_data_iter') must not be None

        Args:
            test_diff (str): filepath of source data
            test_msg (str): filepath of target data
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging
            sem_path (str): filepath of semantic diffs from training set aligned with source diffs by similarity
            out_file (str): filepath of output

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        assert test_diff is not None and out_file is not None

        if batch_size is None:
            raise ValueError("batch_size must be set")

        if os.path.isfile(out_file): os.remove(out_file)

        if sem_path is not None:
            self.sem_score = torch.tensor(
                compute_bleu_score(os.path.join(sem_path, self.sem_diff_default), test_diff))
            self.test_dataset = SemTextDataset(test_diff, test_msg, os.path.join(sem_path, self.sem_diff_default),
                                               self.opt.max_sent_length)

        n_best = self.opt.n_best
        vocab = self.src_vocab

        test_loader = build_dataset_iter(self.test_dataset, vocab, batch_size, gpu=self.gpu, shuffle_batches=False)

        translation_wrapper_builder = TranslationBuilder(self.test_dataset, vocab["tgt"], n_best, len(self.test_dataset.target_texts) > 0)

        # Statistics
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []
        batch_counter = 0
        for batch in test_loader:
            # batch here contains {diff_batch, diff_length, msg_batch, msg_length, sem_batch, sem_length}
            print(f"processing {batch_counter} batch")
            real_batch_size = len(batch['indexes'])
            batch_data = self._process_batch(batch, real_batch_size, sem_path, vocab["tgt"], attn_debug=attn_debug)
            # a batch of results returned from the model is obtained and processed to fit a TranslationWrapper object
            translations = translation_wrapper_builder.from_batch(batch_data, real_batch_size)
            # iter over the objects to build the sentences
            for i, trans in enumerate(translations):
                all_scores += [trans.pred_scores[:n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if test_msg is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred) for pred in trans.pred_sents[:n_best]]
                all_predictions += [n_best_preds]
                with open(out_file, 'a+') as of:
                    of.write('\n'.join(n_best_preds) + '\n')
                    of.flush()

            if self.report_score:
                self.report_manager.report_trans_score('PRED', pred_score_total, pred_words_total)
                if test_msg is not None:
                    self.report_manager.report_trans_score('GOLD', gold_score_total, gold_words_total)
            batch_counter += 1

        self.report_manager.report_trans_eval(out_file, test_msg)
        return all_scores, all_predictions

    def _process_batch(self, batch, batch_size, sem_path, vocab, attn_debug):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object

        """

        max_length = self.opt.max_length
        min_length = self.opt.min_length
        n_best = self.opt.n_best
        return_attention = attn_debug
        beam_size = self.opt.beam_size
        start_token = vocab.vocab[vocabulary.BOS_WORD]
        end_token = vocab.vocab[vocabulary.EOS_WORD]

        with torch.no_grad():
            # Encoder forward.
            src, enc_states, memory_bank, src_lengths = self._run_encoder(batch, batch_size)
            self.model.decoder.init_state(src, memory_bank, enc_states, with_cache=True)
            if sem_path:
                sem, sem_states, sem_bank, sem_lengths = self._run_sem_encoder(batch, batch_size)
                self.sem_decoder.init_state(sem, sem_bank, sem_states, with_cache=True)
            else:
                sem, sem_states, sem_bank, sem_lengths = None, None, None, None

            results = {}
            results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
            results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
            results["attention"] = [[] for _ in range(batch_size)]  # noqa: F812
            results["batch"] = batch
            if batch["tgt_batch"] is not None:
                results["gold_score"] = self._score_target(batch, memory_bank, src_lengths, vocab)
                self.model.decoder.init_state(src, memory_bank, enc_states, with_cache=True)
            else:
                results["gold_score"] = [0] * batch_size

            # Tile states and memory beam_size times.
            self.model.decoder.map_state(lambda state, dim: tile(state, beam_size, dim=dim))
            if isinstance(memory_bank, tuple):
                memory_bank = tuple(tile(x, beam_size, dim=1) for x in memory_bank)
                mb_device = memory_bank[0].device
            else:
                memory_bank = tile(memory_bank, beam_size, dim=1)
                mb_device = memory_bank.device
            memory_lengths = tile(src_lengths, beam_size)

            if sem_path:
                sem_sc = torch.index_select(self.sem_score.to(src.device), 0,
                                            batch["indexes"].to(src.device))  ##simi score
                self.sem_decoder.map_state(lambda state, dim: tile(state, beam_size, dim=dim))
                if isinstance(sem_bank, tuple):
                    sem_bank = tuple(tile(x, beam_size, dim=1) for x in sem_bank)
                else:
                    sem_bank = tile(sem_bank, beam_size, dim=1)
                sem_lengths = tile(sem_lengths, beam_size)
                sem_sc = tile(sem_sc, beam_size).view(-1, 1)
            else:
                sem_sc, sem_lengths, sem_bank = None, None, None
            # beam search aim is to make n hypothesis at the same time: expand the input data [1 x batch x 1] --> [1 x batch*beam x 1], decode, take the 'beam' top values (instead of just one): their indices in mod vocab_size is the n° of token predicted
            top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
            batch_offset = torch.arange(batch_size, dtype=torch.long)
            beam_offset = torch.arange(0, batch_size * beam_size, step=beam_size, dtype=torch.long, device=mb_device)
            alive_seq = torch.full([batch_size * beam_size, 1], start_token, dtype=torch.long, device=mb_device)
            alive_attn = None

            # Give full probability to the first beam on the first step.
            topk_log_probs = (
                torch.tensor([0.0] + [float("-inf")] * (beam_size - 1), device=mb_device).repeat(batch_size))

            # Structure that holds finished hypotheses.
            hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

            for step in range(max_length):
                decoder_input = alive_seq[:, -1].view(1, -1, 1)
                log_probs, attn = self._decode_and_generate(decoder_input, memory_bank,
                                                            memory_lengths=memory_lengths,
                                                            step=step,
                                                            sem_sc=sem_sc, sem_lengths=sem_lengths, sem_bank=sem_bank)

                vocab_size = len(vocab)

                if step < min_length:
                    log_probs[:, end_token] = -1e20

                # Multiply probs by the beam probability.
                log_probs += topk_log_probs.view(-1).unsqueeze(1)

                alpha = 0
                length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

                # Flatten probs into a list of possibilities.
                curr_scores = log_probs / length_penalty
                curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
                # each array are 'beam_size' decoder results of a single batch (concatenated), where each decoder output is the logprobability referred to each token in vocabulary
                topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

                # Recover log probs.
                topk_log_probs = topk_scores * length_penalty

                # Resolve beam origin and true word ids. e.g.[00000] tells that the top5 values were found in the beam 0
                topk_beam_index = torch.tensor(topk_ids.div(vocab_size), dtype=torch.int64)
                topk_ids = topk_ids.fmod(vocab_size)  # specify the token index in the vocabulary

                # Map beam_index to batch_index in the flat representation --> alive_seq has size beam*batch x seq_generation_step
                batch_index = (topk_beam_index + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
                select_indices = batch_index.view(-1)

                # Append last prediction --> the vocabulary token of a beam is appended to it's input sequence which generated it.
                # If in a batch a beam input sequence obtained all topk values (all token different but same beam_index) the alive_seq will be reassigned with all starting input sequence equal but last token generated.
                alive_seq = torch.cat([alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1)

                if return_attention:
                    current_attn = attn.index_select(1, select_indices)
                    if alive_attn is None:
                        alive_attn = current_attn
                    else:
                        alive_attn = alive_attn.index_select(1, select_indices)
                        alive_attn = torch.cat([alive_attn, current_attn], 0)

                is_finished = topk_ids.eq(end_token)
                if step + 1 == max_length:
                    is_finished.fill_(1)

                # Save finished hypotheses.
                if is_finished.any():
                    # Penalize beams that finished.
                    topk_log_probs.masked_fill_(is_finished, -1e10)
                    is_finished = is_finished.to('cpu')
                    top_beam_finished |= is_finished[:, 0].eq(1)
                    predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                    attention = (
                        alive_attn.view(
                            alive_attn.size(0), -1, beam_size, alive_attn.size(-1))
                        if alive_attn is not None else None)
                    non_finished_batch = []
                    for i in range(is_finished.size(0)):
                        b = batch_offset[i]
                        finished_hyp = is_finished[i].nonzero().view(-1)
                        # Store finished hypotheses for this batch.
                        for j in finished_hyp:
                            hypotheses[b].append((
                                topk_scores[i, j],
                                predictions[i, j, 1:],  # Ignore start_token.
                                attention[:, i, j, :memory_lengths[i]]
                                if attention is not None else None))
                        # With top beam finished as end condition we can return n_best hypotheses.
                        if top_beam_finished[i] and len(hypotheses[b]) >= n_best:
                            best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                            for n, (score, pred, attn) in enumerate(best_hyp):
                                if n >= n_best:
                                    break
                                results["scores"][b].append(score)
                                results["predictions"][b].append(pred)
                                results["attention"][b].append(attn if attn is not None else [])
                        else:
                            non_finished_batch.append(i)
                    non_finished = torch.tensor(non_finished_batch)
                    # If all sentences are translated, no need to go further.
                    if len(non_finished) == 0:
                        break

                    # Remove finished batches for the next step.
                    top_beam_finished = top_beam_finished.index_select(0, non_finished)
                    batch_offset = batch_offset.index_select(0, non_finished)
                    non_finished = non_finished.to(topk_ids.device)
                    topk_log_probs = topk_log_probs.index_select(0, non_finished)
                    batch_index = batch_index.index_select(0, non_finished)
                    select_indices = batch_index.view(-1)
                    alive_seq = predictions.index_select(0, non_finished).view(-1, alive_seq.size(-1))
                    if alive_attn is not None:
                        alive_attn = attention.index_select(1, non_finished).view(alive_attn.size(0), -1,
                                                                                  alive_attn.size(-1))

                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(x.index_select(1, select_indices) for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if sem_path:
                    if isinstance(sem_bank, tuple):
                        sem_bank = tuple(x.index_select(1, select_indices) for x in sem_bank)
                    else:
                        sem_bank = sem_bank.index_select(1, select_indices)

                    sem_lengths = sem_lengths.index_select(0, select_indices)
                    sem_sc = sem_sc.index_select(0, select_indices)
                    self.sem_decoder.map_state(lambda state, dim: state.index_select(dim, select_indices))

                self.model.decoder.map_state(lambda state, dim: state.index_select(dim, select_indices))

            return results

    def _run_encoder(self, batch, batch_size):
        src, src_lengths = batch["src_batch"], batch["src_len"]
        enc_states, memory_bank, src_lengths = self.model.encoder(src, src_lengths)
        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), 'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank).long().fill_(memory_bank.size(0))
        return src, enc_states, memory_bank, src_lengths

    def _run_sem_encoder(self, batch, batch_size):
        src = batch["sem_batch"]
        src_lengths = batch["sem_len"]

        src_lengths, rank = src_lengths.sort(descending=True)
        src = src[:, rank, :]
        # shape encoder states (src_length, batch_size, D*Hout) where
        # D is 2 if bidirectional and Hout is the hidden state size
        enc_states, memory_bank, src_lengths = self.model.encoder(src, src_lengths)
        _, recover = rank.sort(descending=False)
        if isinstance(self.model.encoder,TransformerEncoder):
            enc_states = (enc_states[:, recover, :], enc_states[:, recover, :])  # for transformers
        else:
            enc_states = (enc_states[0][:, recover, :], enc_states[1][:, recover, :])

        memory_bank = memory_bank[:, recover, :]
        src_lengths = src_lengths[recover]
        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), 'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank).long().fill_(memory_bank.size(0))
        return src, enc_states, memory_bank, src_lengths

    def _decode_and_generate(self, decoder_input, memory_bank, memory_lengths, step=None, sem_lengths=None, sem_sc=None,
                             sem_bank=None):

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        self.model.decoder.test = 1
        if self.opt.sem_path is not None:
            self.sem_decoder.test = 1
        dec_out, dec_attn = self.model.decoder(
            decoder_input,
            memory_bank,
            memory_lengths=memory_lengths,
            step=step)
        if sem_bank is not None:
            sem_out, sem_attn = self.sem_decoder(
                decoder_input, sem_bank,
                memory_lengths=sem_lengths,
                step=step
            )

        # Generator forward.
        attn = dec_attn["std"]
        log_probs = self.model.generator(dec_out.squeeze(0))
        if sem_sc is not None:
            sem_probs = self.lam_sem * sem_sc.float() * torch.exp(self.model.generator(sem_out.squeeze(0)))
            log_probs = torch.log(torch.tensor(torch.exp(log_probs)) + sem_probs)
        # returns [(batch_size x beam_size) , vocab ] when 1 step
        # or [ tgt_len, batch_size, vocab ] when full sentence

        return log_probs, attn

    def _score_target(self, batch, memory_bank, src_lengths, vocab):
        tgt_in = batch["tgt_batch"][:-1]

        log_probs, attn = self._decode_and_generate(tgt_in, memory_bank, src_lengths)
        tgt_pad = vocab.vocab[vocabulary.PAD_WORD]
        log_probs[:, :, tgt_pad] = 0
        gold = batch["tgt_batch"][1:]
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        return gold_scores