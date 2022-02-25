""" Report manager utility """
from __future__ import print_function

import math
import time
from datetime import datetime
import onmt
import wandb
from evaluate_res import eval_trans_from_files
from onmt.utils.logging import logger

def build_report_manager(opt, action="train"):
    """
    Builder method for the ReportMgr class
    :param opt: dictionary of parameters of the program
    :param action: "train" or "translate" supported actions
    :return: an instance of ReportMgr specialized for the action specified
    """
    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        tensorboard_log_dir = opt.tensorboard_log_dir

        tensorboard_log_dir += datetime.now().strftime("/"+(".".join((opt.models[0].split("/")[-1].split(".")[:-1])
                                                                    if action=="translate" else "")) + "_%b-%d_%H-%M-%S")
        writer = SummaryWriter(tensorboard_log_dir, comment=action)
    else:
        writer = None

    if action == "train":
        report_mgr = ReportMgrTraining(opt.report_every, start_time=-1, tensorboard_writer=writer)
    elif action == "translate":
        report_mgr = ReportMgrTranslation(tensorboard_writer=writer, wandb_run=opt.wandb_run)

    return report_mgr


class ReportMgrTraining(object):
    """
    Report Manager class for Training
    """
    def __init__(self, report_every, start_time=-1., tensorboard_writer=None):
        """
        Init
        :param report_every: (int) Report status every this many sentences
        :param start_time: (float) manually set report start time. Negative values means that you will need to set it later or use `start()`
        :param tensorboard_writer: (:obj:`tensorboard.SummaryWriter`) The TensorBoard Summary writer to use or None
        """

        self.report_every = report_every
        self.progress_step = 0
        self.start_time = start_time
        self.tensorboard_writer = tensorboard_writer

    def start(self):
        self.start_time = time.time()

    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def report_training(self, step, num_steps, learning_rate, teacher_forcing_factor, report_stats, multigpu=False):
        """
        This is the user-defined batch-level training progress report function
        :param step: (int) current step count
        :param num_steps: (int) total number of batches
        :param learning_rate: (float) current learning rate
        :param teacher_forcing_factor: (float) current teacher forcing value
        :param report_stats: (Statistics) old Statistics instance
        :returns report_stats: (Statistics) updated Statistics instance.
        """

        if self.start_time < 0:
            raise ValueError("""ReportMgr needs to be started set 'start_time' or use 'start()'""")

        if step % self.report_every == 0:
            if multigpu:
                report_stats = onmt.utils.Statistics.all_gather_stats(report_stats)
            self._report_training(step, num_steps, learning_rate, teacher_forcing_factor, report_stats)
            self.progress_step += 1
            return onmt.utils.Statistics()
        else:
            return report_stats

    def _report_training(self, step, num_steps, lr, teacher_forcing_factor, report_stats):
        """
        See base class method `ReportMgrBase.report_training`.
        """
        report_stats.output(step, num_steps, lr, self.start_time)
        wandb.log({"tr": step, "tr_lr": lr, "tr_teach": teacher_forcing_factor, "tr_step_ppl": report_stats.ppl(), "tr_step_acc": report_stats.accuracy(), "tr_step_xent": report_stats.xent()}, step)

        # Log the progress using the number of batches on the x-axis.
        self.maybe_log_tensorboard(report_stats, "training", step, lr, teacher_forcing_factor)

    def report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        Report stats of a step
        :param lr: (float) current learning rate
        :param step: (int) current step
        :param train_stats: (Statistics) training stats
        :param valid_stats: (Statistics) validation stats
        """

        self._report_step(lr, step, train_stats=train_stats, valid_stats=valid_stats)

    def _report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        See base class method `ReportMgrBase.report_step`.
        """
        if train_stats is not None:
            self.log('Train perplexity: %g' % train_stats.ppl())
            self.log('Train accuracy: %g' % train_stats.accuracy())
            wandb.log({"tr_step": step, "tr_step_lr": lr, "tr_step_ppl": train_stats.ppl(), "tr_step_acc": train_stats.accuracy(), "tr_step_xent": train_stats.xent()}, step)

            self.maybe_log_tensorboard(train_stats, "train_step", step, lr)

        if valid_stats is not None:
            self.log('Validation perplexity: %g' % valid_stats.ppl())
            self.log('Validation accuracy: %g' % valid_stats.accuracy())
            wandb.log({"val_step": step, "val_step_lr": lr, "val_step_ppl": valid_stats.ppl(), "val_step_acc": valid_stats.accuracy(), "val_step_xent": valid_stats.xent()}, step)

            self.maybe_log_tensorboard(valid_stats, "valid_step", step, lr)

    def maybe_log_tensorboard(self, stats, prefix, step, learning_rate=None, teacher_forcing_factor=None):
        if self.tensorboard_writer is not None:
            stats.log_tensorboard(prefix, self.tensorboard_writer, step, learning_rate, teacher_forcing_factor)


class ReportMgrTranslation(object):
    def __init__(self, tensorboard_writer=None, wandb_run=None):
        """
        A report manager that writes statistics on standard output as well as (optionally) TensorBoard, W&B
        :param tensorboard_writer: (:obj:`tensorboard.SummaryWriter`) The TensorBoard Summary writer to use or None
        :param wandb_run: the identifier of W&B run
        """
        self.tensorboard_writer = tensorboard_writer
        self.wandb_run = wandb.Api().run(wandb_run) if wandb_run is not None else None

    def report_model_details(self, model_stats=None, semantic=None):
        logger.info("Translation {} semantics".format("WITH" if semantic else "WITHOUT"))
        if model_stats is not None:
            train, val = model_stats
            t_acc, t_ppl, t_xent = train.accuracy(), train.ppl(), train.xent()
            v_acc, v_ppl, v_xent = val.accuracy(), val.ppl(), val.xent()
            logger.info(f"Model stats: Training: {t_acc} acc, {t_ppl} ppl, {t_xent} xent"
                                   f"  Validation: {v_acc} acc, {v_ppl} ppl, {v_xent} xent")
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalars("translate/accuracy", {"train": t_acc, "val": v_acc}, 0)
                self.tensorboard_writer.add_scalars("translate/ppl", {"train": t_ppl, "val": v_ppl}, 0)
                self.tensorboard_writer.add_scalars("translate/xent", {"train": t_xent, "val": v_xent}, 0)
                self.tensorboard_writer.add_scalar("translate/use_semantic", int(semantic), 0)

    def report_trans_eval(self, translations_file, targets_file):
        results = eval_trans_from_files(translations_file, targets_file)
        meteor_score = results["Meteor"]
        rouge_score = results["Rouge"]
        bert_p_score, bert_r_score, bert_f1_score = results["BertScore"]
        bleu_score, bleu_ngrams = results["Bleu"]

        # Console
        logger.info(f"TEST SET SCORES\n"
                    f"Meteor: {meteor_score}\n"
                    f"Rouge: {rouge_score}\n"
                    f"Bert p, r, f1: {bert_p_score, bert_r_score, bert_f1_score}\n"
                    f"Bleu: {bleu_ngrams}\n"
                    f"Bleu mean {bleu_score}")
        # Weights and Bias
        if self.wandb_run is not None:
            self.wandb_run.summary["Meteor"] = meteor_score
            self.wandb_run.summary["Rouge"] = rouge_score
            self.wandb_run.summary["Bert_p"] = bert_p_score
            self.wandb_run.summary["Bert_r"] = bert_r_score
            self.wandb_run.summary["Bert_f1"] = bert_f1_score
            self.wandb_run.summary["Bleu"] = bleu_score
            self.wandb_run.summary["Bleu_1"] = bleu_ngrams[0]
            self.wandb_run.summary["Bleu_2"] = bleu_ngrams[1]
            self.wandb_run.summary["Bleu_3"] = bleu_ngrams[2]
            self.wandb_run.summary["Bleu_4"] = bleu_ngrams[3]
            self.wandb_run.summary.update()
        # Tensorboard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar("translate/nist", rouge_score, 0)
            self.tensorboard_writer.add_scalar("translate/meteor", meteor_score, 0)
            self.tensorboard_writer.add_scalar("translate/bleu", bleu_score, 0)
            self.tensorboard_writer.add_scalar("translate/bleu", bleu_ngrams[0], 1)
            self.tensorboard_writer.add_scalar("translate/bleu", bleu_ngrams[1], 2)
            self.tensorboard_writer.add_scalar("translate/bleu", bleu_ngrams[2], 3)
            self.tensorboard_writer.add_scalar("translate/bleu", bleu_ngrams[3], 4)

    def report_trans_score(self, name, score_total, words_total):
        if words_total == 0:
            msg = "%s No words predicted" % (name,)
        else:
            msg = ("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
                name, score_total / words_total, name, math.exp(-score_total / words_total)))
        logger.info(msg)


