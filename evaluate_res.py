import numpy as np
from bert_score import BERTScorer
from nltk.translate.meteor_score import single_meteor_score
from torchtext.data import bleu_score

from evaluation.rouge.rouge import Rouge


def _g_mean(x):
    a = np.log(x)
    return np.exp(a.mean())


def get_bleu(translations, references):
    """
    Compute the bleu score
    :param translations: list of hypothesis sentences splitted
    :param references: list of targets sentences splitted
    :return: mean, array of 4 ngrams
    """
    references = [[r] for r in references]
    bleu1 = bleu_score(translations, references, max_n=1, weights=[0.25])
    bleu2 = bleu_score(translations, references, max_n=2, weights=[0.25, 0.25])
    bleu3 = bleu_score(translations, references, max_n=3, weights=[0.25, 0.25, 0.25])
    bleu4 = bleu_score(translations, references, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
    sum_ref = np.sum([len(l) for l in references])
    sum_pred = np.sum([len(l) for l in translations])
    brevity_penalty = 1 if sum_ref <= sum_pred else np.e ** (1 - (sum_ref / sum_pred))
    g_mean = _g_mean([bleu1, bleu2, bleu3, bleu4]) * brevity_penalty
    return g_mean, [bleu1, bleu2, bleu3, bleu4]

def evaluate_translations(translations, references, metrics = None):
    """
    Evaluate translations
    :param translations: list of hypothesis sentences splitted
    :param references: list of targets sentences splitted
    :param metrics: list of metrics to evaluate if available
    :return: a dictionary with computed values for metrics specified or else all available
    """
    t_joined = [" ".join(r) for r in references]
    r_joined = [" ".join(t) for t in translations]
    t_key = {k: [v.strip().lower()] for k, v in enumerate(t_joined)}
    r_key = {k: [v.strip().lower()] for k, v in enumerate(r_joined)}
    results = {}

    # Compute scores
    if metrics is None or "Rouge" in metrics:
        rouge_score, _ = Rouge().compute_score(r_key, t_key)
        results["Rouge"] = rouge_score

    if metrics is None or "Meteor" in metrics:
        meteor_score = np.mean([single_meteor_score(r, p) for r, p in zip(references, translations)])
        results["Meteor"] = meteor_score

    if metrics is None or "BertScore" in metrics:
        precision, recall, f1 = BERTScorer(lang="en", rescale_with_baseline=True).score(t_joined, r_joined)
        results["BertScore"] = float(precision.mean()), float(recall.mean()), float(f1.mean())

    if metrics is None or "Bleu" in metrics:
        results["Bleu"] = get_bleu(translations, references)

    return results

def eval_trans_from_files(translations_file, references_file):
    """
    Evaluate translations from files
    :param translations_file: file path of hypothesis sentences
    :param references_file: file path of targets sentences
    :return: a dictionary with computed values for all available metrics
    """
    with open(translations_file, 'r') as file:
        lines = file.readlines()
        translations = [t.strip().lower().split(" ") for t in lines]
    with open(references_file, 'r') as file:
        lines = file.readlines()
        references = [r.strip().lower().split(" ") for r in lines]

    return evaluate_translations(translations, references)


def main(hyp, ref):
    results = eval_trans_from_files(hyp, ref)
    print(f"Meteor: {results['Meteor']}")
    print(f"Rouge: {results['Rouge']}")
    print(f"Bert precision: {results['BertScore'][0]}, Bert recall: {results['BertScore'][1]}, Bert F1: {results['BertScore'][2]}")
    print(f"Bleu: {results['Bleu'][1]}, Bleu mean {results['Bleu'][0]}")


if __name__ == '__main__':
    pred_CoRec = "data/output/1000test.out"
    pred_ptg = "result/ptg/cleaned.test.msg"
    pred_nngen = "result/nngen/cleaned.test.msg"
    ref = "data/top1000/cleaned.test.msg"

    print("PTG")
    main(pred_ptg, ref)
    print("NNGEN")
    main(pred_nngen, ref)
    print("CoRec")
    main(pred_CoRec, ref)