from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import numpy as np

from onmt.hashes.smooth import compute_bleu_score


def main(hyp, ref):
    with open(hyp, 'r') as r:
        hypothesis = r.readlines()
        res = {k: [v.strip().lower()] for k, v in enumerate(hypothesis)}
    with open(ref, 'r') as r:
        references = r.readlines()
        tgt = {k: [v.strip().lower()] for k, v in enumerate(references)}

    #TODO fix script for Meteor and check against their reported results in table 4.5
    score_Meteor, scores_Meteor = Meteor().compute_score(tgt, res)
    print("Meteor: %s" % score_Meteor)

    score_Rouge, scores_Rouge = Rouge().compute_score(tgt, res)
    print("ROUGE: %s" % score_Rouge)

    #TODO Find correct bleu implementation used for the evaluation
    print("Bleu: %s" % np.mean(compute_bleu_score(ref,pred)))


if __name__ == '__main__':
    # put your path here to not import their results in github or whatever 
    pred = "../result/CoRec/cleaned.test.msg"
    ref = "../result/ref/cleaned.test.msg"
    main(pred, ref)

#TODO once found the right evaluation scripts we can try to understand how
#       they obtained such file they put in the folder or results: maybe we will discover that the results alredy obtained
#       evaluated in this way has same scores and we are good to proceed in that case
