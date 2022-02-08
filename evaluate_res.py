from onmt.evaluation.pycocoevalcap.meteor.meteor import Meteor
from onmt.evaluation.pycocoevalcap.rouge.rouge import Rouge
from onmt.evaluation.pycocoevalcap.bleu.bleu import Bleu

def main(hyp, ref):
    with open(hyp, 'r') as r:
        hypothesis = r.readlines()
        res = {k: [v.strip().lower()] for k, v in enumerate(hypothesis)}
    with open(ref, 'r') as r:
        references = r.readlines()
        tgt = {k: [v.strip().lower()] for k, v in enumerate(references)}

    score_Meteor, scores_Meteor = Meteor().compute_score(tgt, res)
    print("Meteor: %s" % score_Meteor)

    score_Rouge, scores_Rouge = Rouge().compute_score(tgt, res)
    print("ROUGE: %s" % score_Rouge)

    gmean, score_Bleu, scores_Bleu = Bleu().compute_score(tgt, res)
    print(f"Bleu: {score_Bleu}, Bleu mean {gmean}")


if __name__ == '__main__':
    pred_CoRec = "data/output/1000test.out"
    #pred_ptg = "result/ptg/cleaned.test.msg"
    #pred_nngen = "result/nngen/cleaned.test.msg"
    ref = "data/top1000/cleaned.test.msg"

    #print("PTG")
    #main(pred_ptg, ref)
    #print("NNGEN")
    #main(pred_nngen, ref)
    #print("CoRec")
    main(pred_CoRec, ref)