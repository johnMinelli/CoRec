"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""
import torch

import onmt.utils
import onmt

from onmt.helpers.report_manager import build_report_manager
from onmt.utils.logging import logger
from onmt.utils.loss import build_loss_compute

def build_trainer(opt, model, vocab, optim, model_saver):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        vocab (dict): dictionary with embedding of all tokens
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    train_loss = build_loss_compute(model, vocab, opt)
    valid_loss = build_loss_compute(model, vocab, opt, train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    grad_accum_count = opt.accum_count
    gpu_verbose_level = opt.gpu_verbose_level

    report_manager = build_report_manager(opt)
    trainer = Trainer(model, train_loss, valid_loss, optim, trunc_size,
                           shard_size, grad_accum_count,
                           gpu_verbose_level, report_manager,
                           model_saver=model_saver)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, grad_accum_count=1,
                 gpu_verbose_level=0, report_manager=None, model_saver=None):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.grad_accum_count = grad_accum_count
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver

        assert grad_accum_count > 0
        if grad_accum_count > 1:
            assert (self.trunc_size == 0), """To enable accumulated gradients, you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter_fct, valid_iter_fct, train_steps, valid_steps):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...')

        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        train_iter = train_iter_fct()

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:

            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                logger.info(f"Batch: {i} accum: {accum}") if self.gpu_verbose_level > 1 else None
                true_batchs.append(batch)

                normalization += batch[0][1].size(0)

                accum += 1
                if accum == self.grad_accum_count:
                    reduce_counter += 1
                    logger.info(f"Reduce_counter: {reduce_counter} n_minibatch {len(true_batchs)}") if self.gpu_verbose_level > 0 else None

                    self._gradient_accumulation(true_batchs, normalization, total_stats, report_stats, step)
                    report_stats = self._maybe_report_training(step, train_steps, self.optim.learning_rate, report_stats)

                    true_batchs = []
                    accum = 0
                    normalization = 0
                    if (step % valid_steps == 0):
                        logger.info(f'Validate step {step}') if self.gpu_verbose_level > 0 else None
                        valid_iter = valid_iter_fct()
                        valid_stats = self.validate(valid_iter)
                        logger.info(f'Report stat step {step}') if self.gpu_verbose_level > 0 else None
                        self._report_step(self.optim.learning_rate, step, valid_stats=valid_stats)

                    self._maybe_save(step)
                    step += 1
                    if step > train_steps:
                        break
            logger.info(f'Epoch completed at step {step}') if self.gpu_verbose_level > 0 else None
            train_iter = train_iter_fct()

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = onmt.utils.Statistics()
        # why it doesn't use the with torch.no_grad():
        for batch in valid_iter:
            src, source_lengths = batch[0]
            tgt = batch[1]

            # F-prop through the model.
            outputs, attention = self.model(src, tgt, source_lengths)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(batch, outputs, attention)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def _gradient_accumulation(self, true_batchs, normalization, total_stats, report_stats, step):  #+++ step
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            # if samples are not padded correctly with same size an error will rise
            src, source_lengths = batch[0]
            tgt_outer = batch[1]

            target_size = tgt_outer.size(0)
            report_stats.n_src_words += source_lengths.sum().item()

            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()

                self.model.set_step(step)
                outputs, attention = self.model(src, tgt, source_lengths)

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(batch, outputs, attention, j, trunc_size, self.shard_size, normalization)  # probably normalization value is wrong
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    self.optim.step()

                if self.model.decoder.state is not None:
                    self.model.decoder.detach_state()

        # in case of multi step gradient accumulation, update only after accum batches
        if self.grad_accum_count > 1:
            self.optim.step()

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(step, num_steps, learning_rate, report_stats)

    def _report_step(self, learning_rate, step, train_stats=None, valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(learning_rate, step, train_stats=train_stats, valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
