"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""
import math
import random

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
    train_loss = build_loss_compute(model, vocab["tgt"], opt)
    valid_loss = build_loss_compute(model, vocab["tgt"], opt, train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    grad_accum_count = opt.accum_count
    gpu_verbose_level = opt.gpu_verbose_level

    # From scheduled_sampling
    ###
    norm_method = opt.normalization
    twopass = opt.decoder_type == 'transformer'
    sampling_type = opt.sampling_type
    scheduled_sampling_decay = opt.scheduled_sampling_decay
    scheduled_sampling_k = opt.scheduled_sampling_k
    scheduled_sampling_c = opt.scheduled_sampling_c
    scheduled_sampling_limit = opt.scheduled_sampling_limit
    mixture_type = opt.mixture_type
    topk_value = opt.topk_value
    peeling_back = opt.peeling_back
    passone_nograd = (not opt.transformer_passone) or \
                     (opt.transformer_passone and opt.transformer_passone == 'nograd')
    scheduled_activation = opt.transformer_scheduled_activation
    scheduled_softmax_alpha = opt.transformer_scheduled_alpha
    ###

    report_manager = build_report_manager(opt, "train")
    trainer = Trainer(model, train_loss, valid_loss, optim, trunc_size,
                      shard_size, grad_accum_count,
                      gpu_verbose_level, report_manager,
                      model_saver=model_saver) if opt.decoder_type != 'transformer' else \
        TransformerTrainer(model, train_loss, valid_loss, optim, trunc_size,
                           shard_size, norm_method,
                           grad_accum_count,
                           gpu_verbose_level, report_manager,
                           model_saver=model_saver,
                           sampling_type=sampling_type,
                           scheduled_sampling_decay=scheduled_sampling_decay,
                           scheduled_sampling_k=scheduled_sampling_k,
                           scheduled_sampling_c=scheduled_sampling_c,
                           scheduled_sampling_limit=scheduled_sampling_limit,
                           mixture_type=mixture_type,
                           topk_value=topk_value,
                           peeling_back=peeling_back,
                           twopass=twopass,
                           passone_nograd=passone_nograd,
                           scheduled_activation=scheduled_activation,
                           scheduled_softmax_alpha=scheduled_softmax_alpha)
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
            assert (
                    self.trunc_size == 0), """To enable accumulated gradients, you must disable target sequence truncating."""

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

                normalization += batch["src_len"].size(0)

                accum += 1
                if accum == self.grad_accum_count:
                    reduce_counter += 1
                    logger.info(
                        f"Reduce_counter: {reduce_counter} n_minibatch {len(true_batchs)}") if self.gpu_verbose_level > 0 else None

                    self._gradient_accumulation(true_batchs, normalization, total_stats, report_stats, step)
                    report_stats = self._maybe_report_training(step, train_steps, self.optim.learning_rate,
                                                               report_stats)

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

        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                src = batch["src_batch"]
                source_lengths = batch["src_len"]
                tgt = batch["tgt_batch"]
                tgt_lengths = batch["tgt_len"]

                # F-prop through the model.
                outputs, attention = self.model(src, tgt, source_lengths)

                # Compute loss.
                batch_stats = self.valid_loss.monolithic_compute_loss(batch, outputs, attention)

                # Update statistics.
                stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def _gradient_accumulation(self, true_batchs, normalization, total_stats, report_stats, step):  # +++ step
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            # if samples are not padded correctly with same size an error will rise
            src = batch["src_batch"]
            source_lengths = batch["src_len"]
            tgt_outer = batch["tgt_batch"]
            tgt_lengths = batch["tgt_len"]

            target_size = tgt_outer.size(0)
            report_stats.n_src_words += source_lengths.sum().item()

            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            for j in range(0, target_size - 1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()

                self.model.set_step(step)
                outputs, attention = self.model(src, tgt, source_lengths)

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(batch, outputs, attention, j, trunc_size,
                                                                   self.shard_size,
                                                                   normalization)  # probably normalization value is wrong
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
            return self.report_manager.report_step(learning_rate, step, train_stats=train_stats,
                                                   valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)


class TransformerTrainer(Trainer):
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
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, valid_loss, optim, trunc_size=0, shard_size=32, norm_method="sents",
                 grad_accum_count=1, gpu_verbose_level=0, report_manager=None, model_saver=None,
                 sampling_type="teacher_forcing", scheduled_sampling_decay="exp", scheduled_sampling_k=1.0,
                 scheduled_sampling_c=1.0, scheduled_sampling_limit=0.0, mixture_type='none', topk_value=1,
                 peeling_back='none', twopass=False, passone_nograd='nograd', scheduled_activation='softmax',
                 scheduled_softmax_alpha='1.0'):
        super().__init__(model, train_loss, valid_loss, optim, trunc_size, shard_size, grad_accum_count,
                         gpu_verbose_level, report_manager, model_saver)
        self.model = model
        self.train_loss = train_loss
        self._valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self._norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver
        self._sampling_type = sampling_type
        self._scheduled_sampling_decay = scheduled_sampling_decay
        self._scheduled_sampling_k = scheduled_sampling_k
        self._scheduled_sampling_c = scheduled_sampling_c
        self._scheduled_sampling_limit = scheduled_sampling_limit
        self._mixture_type = mixture_type
        self._k = topk_value
        self._peeling_back = peeling_back
        self._twopass = twopass
        self._passone_nograd = passone_nograd
        if scheduled_activation == "sparsemax":
            self._scheduled_activation_function = onmt.modules.sparse_activations.Sparsemax(dim=-1)
        # elif scheduled_activation == "gumbel":
        #    self._scheduled_activation_function = onmt.modules.softmax_extended.GumbelSoftmax(dim=-1,
        #                                                                                      alpha=scheduled_softmax_alpha)
        # elif scheduled_activation == "softmax_temp":
        #    self._scheduled_activation_function = onmt.modules.softmax_extended.SoftmaxWithTemperature(dim=-1,
        #                                                                                               alpha=scheduled_softmax_alpha)
        else:
            self._scheduled_activation_function = torch.nn.Softmax(dim=-1)

        assert grad_accum_count == 1  # disable grad accumulation

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

            # there should be only one loop
            for i, batch in enumerate(train_iter):
                logger.info(f"Batch: {i} accum: {accum}") if self.gpu_verbose_level > 1 else None

                true_batchs.append(batch)

                if self._norm_method == "tokens":
                    num_tokens = batch.tgt[1:].ne(
                        self.train_loss.padding_idx).sum()
                    normalization += num_tokens.item()
                else:
                    normalization += batch["src_len"].size(0)
                accum += 1
                if accum == self.grad_accum_count:

                    start_decay = 4500
                    batch_teacher_forcing_ratio = \
                        self._calc_teacher_forcing_ratio(step, start_decay)

                    # print('TRANSF_GRAD: step: ', step)
                    if step % 200 == 0:
                        print(f"batch_teacher_forcing_ratio: {batch_teacher_forcing_ratio}")

                    self._train_batch(
                        batch, normalization, total_stats, report_stats,
                        batch_teacher_forcing_ratio, start_decay, step
                    )

                    report_stats = self._maybe_report_training(
                        step, train_steps,
                        self.optim.learning_rate,
                        report_stats)

                    true_batchs = []
                    accum = 0
                    normalization = 0
                    if step % valid_steps == 0:
                        if self.gpu_verbose_level > 0:
                            logger.info('GpuRank: validate step %d'
                                        % step)
                        valid_iter = valid_iter_fct()
                        valid_stats = self.validate(valid_iter)
                        if self.gpu_verbose_level > 0:
                            logger.info('GpuRank: gather valid stat \
                                        step %d' % step)
                        if self.gpu_verbose_level > 0:
                            logger.info('GpuRank: report stat step %d'
                                        % step)
                        self._report_step(self.optim.learning_rate, step, valid_stats=valid_stats)

                    self._maybe_save(step)
                    step += 1
                    if step > train_steps:
                        break
            if self.gpu_verbose_level > 0:
                logger.info('GpuRank : we completed an epoch \
                            at step %d' % step)

        return total_stats

    def _calc_teacher_forcing_ratio(self, step, start_decay):
        if self._sampling_type == "teacher_forcing" or step <= start_decay:
            return 1.0
        elif self._sampling_type == "scheduled":  # scheduled sampling
            # linear decay
            scheduled_ratio = self._scheduled_sampling_k - self._scheduled_sampling_c * (step - start_decay)
            scheduled_ratio = max(self._scheduled_sampling_limit, scheduled_ratio)
            return scheduled_ratio
        else:  # always sample from the model predictions
            return 0.0

    def _train_batch(self, batch, normalization, total_stats, report_stats,
                     teacher_forcing_ratio, start_decay, step=None ):
        target_size = batch["tgt_batch"].size(0)
        trunc_size = self.trunc_size if self.trunc_size else target_size

        src = batch["src_batch"]
        src_lengths = batch["src_len"]

        tgt_outer = batch["tgt_batch"]

        dec_state = None
        emb_weights = None
        top_k_tgt = None
        tf_gate_value = None

        for j in range(0, target_size - 1, trunc_size):
            # 1. Create truncated target.
            tgt = tgt_outer[j: j + trunc_size]

            # Two pass process
            tf_tgt_section = round(target_size * teacher_forcing_ratio)
            if tf_tgt_section >= target_size:
                # The standard model
                outputs, attns = self.model(src, tgt, src_lengths)
            else:
                tgt = tgt[:-1]

                # 1. Go through the encoder
                enc_state, memory_bank, lengths = \
                    self.model.encoder(src, src_lengths)
                self.model.decoder.init_state(src, memory_bank, enc_state)

                # This part can be with grad or no_grad
                if self._passone_nograd:
                    with torch.no_grad():
                        outputs, attns = self.model.decoder(tgt, memory_bank,
                                                            memory_lengths=lengths)
                        logits = self.model.generator[0](outputs)

                else:
                    outputs, attns = self.model.decoder(tgt, memory_bank,
                                                        memory_lengths=lengths)

                    logits = self.model.generator[0](outputs)

                # 2. Get the embeddings from the model predictions
                if self._mixture_type and 'topk' in self._mixture_type:
                    k = self._k
                    emb_weights, top_k_tgt = logits.topk(k, dim=-1)

                    # Needed for getting the embeddings
                    top_k_tgt = top_k_tgt.unsqueeze(-2)

                    # k_embs: batch x k x emb size
                    k_embs = self.model.decoder.embeddings(top_k_tgt, step=0).transpose(2, 3)
                    # weights: batch x sequence length x k x 1
                    # Normalize the weights
                    emb_weights /= emb_weights.sum(dim=-1).unsqueeze(2)
                    weights = emb_weights.unsqueeze(3)
                    emb_size = k_embs.shape[2]
                    embeddings = self.model.decoder.embeddings(top_k_tgt, step=0)
                    model_prediction_emb = torch.bmm(k_embs.view(-1, emb_size, k),
                                                     weights.view(-1, k, 1))  # .transpose(0, 1)
                    model_prediction_emb = model_prediction_emb.view(batch["src_len"].size(0), -1, emb_size).transpose(
                        0, 1)
                elif self._mixture_type and 'all' in self._mixture_type:
                    logits = self._scheduled_activation_function(logits)

                    # weights = logits
                    # Get the indices of all words in the vocabulary
                    ind = torch.cuda.LongTensor([i for i in range(logits.shape[2])])
                    # We need this format of the indices to ge tht embeddings from the decoder
                    ind = ind.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    embeddings = self.model.decoder.embeddings(ind, step=0)[0][0]

                    # The predicted embedding is the weighted sum of the words in the vocabulary
                    model_prediction_emb = torch.matmul(logits, embeddings)
                else:
                    # Just get the argmax from the model predictions
                    logits = self.model.generator[1](logits)
                    model_predictions = logits.argmax(dim=2).unsqueeze(2)
                    model_prediction_emb = self.model.decoder.embeddings(model_predictions)

                # Get the embeddings of the gold target sequence.
                tgt_emb = self.model.decoder.embeddings(tgt)

                # 3. Combine the gold target with the model predictions
                if self._peeling_back == 'strict':
                    # Combine the two sequences with peelingback
                    # First part from the gold, second part from the model predictions
                    tf_tgt_emb = torch.cat((tgt_emb[:tf_tgt_section],
                                            model_prediction_emb[tf_tgt_section:]))
                else:
                    # Use scheduled sampling - on each step decide
                    # whether to use teacher forcing or model predictions.
                    tf_tgt_emb = [tgt_emb[i].unsqueeze(0) \
                                      if random.random() <= teacher_forcing_ratio else \
                                      model_prediction_emb[i].unsqueeze(0) for i in range(target_size - 1)]
                    # tf_tgt_emb.append(tgt_emb[-1].unsqueeze(0))
                    tf_tgt_emb = torch.cat((tf_tgt_emb), dim=0)
                # Rerun the forward pass with the new target context
                outputs, attns = self.model.decoder(tgt, memory_bank,
                                                    memory_lengths=lengths, step=None, tf_emb=tf_tgt_emb)

            # 3. Compute loss in shards for memory efficiency.
            # print('memory sizes:', trunc_size, self.shard_size, len(batch), outputs.shape)
            # print('before loss:', outputs.shape)
            # print('batch:', batch)
            batch_stats = self.train_loss.sharded_compute_loss(
                batch, outputs, attns, j,
                trunc_size, self.shard_size, normalization)
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)
            # 4. Update the parameters and statistics.

            self.optim.step()

            # If truncated, don't backprop fully.
            if dec_state is not None:
                dec_state.detach()

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _norm(self, batch):
        if self._norm_method == "tokens":
            norm = batch.tgt[1:].ne(self.train_loss.padding_idx).sum()
        else:
            norm = batch.batch_size
        return norm
