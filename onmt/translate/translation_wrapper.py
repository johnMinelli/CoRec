""" Translation main class """
from __future__ import unicode_literals, print_function

import torch

from onmt.hashes.smooth import get_bleu
from onmt.inputters.vocabulary import EOS_WORD, UNK


class TranslationBuilder(object):
    """
    Build a word-based translation TranslationWrapper from the batch output
    of translator and the underlying dictionaries.

    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`

    Args:
       data (DataSet):
       fields (dict of Fields): data fields
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
       has_tgt (bool): will the batch have gold targets
    """

    def __init__(self, dataset, vocab, n_best=1, has_tgt=False):
        self.dataset = dataset
        self.vocab = vocab
        self.n_best = n_best
        self.has_tgt = has_tgt

    def _build_target_tokens(self, pred):
        tokens = []
        for index in pred:
            if index < len(self.vocab):
                tokens.append(self.vocab.lookup_token(index))
            else:
                tokens.append(" ")
            if tokens[-1] == EOS_WORD:
                tokens = tokens[:-1]
                break
        return tokens

    def from_batch(self, translation_batch, batch_size):
        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) == len(translation_batch["predictions"]))

        preds, pred_score, attn, gold_score, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["gold_score"],
                        batch["indexes"]),
                    key=lambda x: x[-1])))

        # Sorting
        inds, perm = torch.sort(batch["indexes"].to(batch["src_batch"].device))
        src = batch["src_batch"].data.index_select(1, perm)

        if self.has_tgt:
            tgt = batch["tgt_batch"].index_select(1, perm)
        else:
            tgt = None

        translations = []
        for b in range(batch_size):
            src_raw = self.dataset[inds[b]][0]
            pred_sents = [self._build_target_tokens(preds[b][n]) for n in range(self.n_best)]
            gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(tgt[1:, b] if tgt is not None else None)

            translation = TranslationWrapper(src[:, b] if src is not None else None,
                                             src_raw, pred_sents,
                                             attn[b], pred_score[b], gold_sent,
                                             gold_score[b])
            translations.append(translation)

        return translations


class TranslationWrapper(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number, rouge=None):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
            output += ("BLEU SCORE: {:.4f}\n".format(get_bleu(pred_sent, tgt_sent)))
            if rouge: output += f"ROUGE SCORE: {rouge.score(pred_sent, tgt_sent)}\n"
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
