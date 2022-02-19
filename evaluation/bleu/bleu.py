#!/usr/bin/env python
#
# File Name : bleu.py
#
# Description : Wrapper for BLEU scorer.
#
# Creation Date : 06-01-2015
# Last Modified : Thu 19 Mar 2015 09:13:28 PM PDT
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>
import numpy as np

from .bleu_scorer import BleuScorer

class Bleu:
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):

        def g_mean(x):
            a = np.log(x)
            return np.exp(a.mean())

        assert(list(gts.keys()) == list(res.keys()))
        imgIds = list(gts.keys())

        bleu_scorer = BleuScorer(n=self._n)
        for idx in imgIds:
            hypo = res[idx]
            ref = gts[idx]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        # score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        # score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        sumg = 0
        for i, l in gts.items():
            sumg += len(l[0].split())
        sumr = 0
        for i, l in res.items():
            sumr += len(l[0].split())
        brevity_penalty = 1 if sumg <= sumr else np.e ** (1 - ( sumg / sumr))
        gmean = g_mean(score) * brevity_penalty

        return gmean, score, scores

    def method(self):
        return "Bleu"
