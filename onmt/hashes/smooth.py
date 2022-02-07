# coding=utf-8
import math
from onmt.utils.misc import read_file
import numpy as np
from onmt.evaluation.pycocoevalcap.bleu.bleu import Bleu
import  nltk.translate.bleu_score as bleu

def get_bleu(reference, hypothesis):
    gts = {0: [reference]}
    res = {0: [hypothesis]}
    score_Bleu, scores_Bleu = Bleu(1).compute_score(gts, res)
    return np.around(np.mean(score_Bleu), 4)


def write_file(filename, data):
    with open(filename, 'a+') as f:
        for i in data:
            f.write("%.4f\n" % i)


def compute_bleu_score(sem_diff_path, test_diff_path):
    test_diffs = read_file(test_diff_path)
    sem_diffs = read_file(sem_diff_path)
    sem_scores = []

    for sem, test in zip(sem_diffs, test_diffs):
        # bleu_score_sem = bleu.sentence_bleu([sem.split()], test.split())
        bleu_score_sem = get_bleu(sem, test)

        sem_scores.append(bleu_score_sem)

    # write_file("data/new/syn_bleu.score", syn_scores)
    # write_file("data/new/sem_bleu.score", sem_scores)
    return sem_scores

if __name__ == '__main__':
    syn_diff_path = "data/new/test.syn.diff"
    sem_diff_path = "data/new/test_tf.sem.diff"
    compute_bleu_score(syn_diff_path, sem_diff_path)
