#!/usr/bin/env python
"""
    Training on a single process
"""

import configargparse

import os
import random
import torch

import onmt.opts as opts
from onmt.helpers.model_saver import build_model_saver
from onmt.helpers.trainer import build_trainer
from onmt.helpers.model_builder import build_model
from onmt.inputters.input_aux import load_dataset, build_dataset_iter, load_vocab
from onmt.utils.logging import init_logger, logger
from onmt.utils.optimizers import build_optim


def _count_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    return n_params, enc, dec


def training_opt_parsing(opt, device_id):
    if opt.word_vec_size != -1:
        opt.src_word_vec_size = opt.word_vec_size
        opt.tgt_word_vec_size = opt.word_vec_size

    if opt.layers != -1:
        opt.enc_layers = opt.layers
        opt.dec_layers = opt.layers

    if opt.rnn_size != -1:
        opt.enc_rnn_size = opt.rnn_size
        opt.dec_rnn_size = opt.rnn_size

    opt.brnn = (opt.encoder_type == "brnn")

    if torch.cuda.is_available() and not opt.gpu:
        logger.info("WARNING: You have a CUDA device, should run with -gpu")

    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        # for torchtext random call (shuffled iterator), in a multi gpu env ensures datasets are read in the same order
        random.seed(opt.seed)
        # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        if opt.seed > 0:
            # These ensure same initialization in multi gpu mode
            torch.cuda.manual_seed(opt.seed)

    return opt


def main(opt, device_id):
    opt = training_opt_parsing(opt, device_id)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from, map_location=lambda storage, loc: storage)

        # overwrite options with the ones from checkpoint
        dummy_parser = configargparse.ArgumentParser()
        opts.model_opts(dummy_parser)
        default_opt = dummy_parser.parse_known_args([])[0]

        model_opt = default_opt
        model_opt.__dict__.update(checkpoint['opt'].__dict__)

        # copy the following new parameters
        model_opt.save_model = opt.save_model
        model_opt.save_checkpoint_steps = opt.save_checkpoint_steps
        model_opt.keep_checkpoint = opt.keep_checkpoint
    else:
        checkpoint = None
        model_opt = opt

    # Load fields generated from preprocess phase.
    vocab = load_vocab(opt.data + '.vocab.pt', checkpoint)
    logger.info(' * vocabulary size. source = %d' % len(vocab))

    # Build model.
    model = build_model(model_opt, vocab, opt.gpu, checkpoint)
    n_params, enc, dec = _count_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)

    # Build optimizer.
    optim = build_optim(model, opt, checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, model, vocab, optim)

    trainer = build_trainer(opt, model, vocab, optim, model_saver)

    def train_iter_fct():
        return build_dataset_iter(load_dataset("train", opt), vocab, opt.batch_size)

    def valid_iter_fct():
        return build_dataset_iter(load_dataset("valid", opt), vocab, opt.valid_batch_size)

    # Do training.
    if opt.gpu:
        logger.info('Starting training on GPU')
    else:
        logger.info('Starting training on CPU, could be very slow')
    trainer.train(train_iter_fct, valid_iter_fct, opt.train_steps, opt.valid_steps)
    torch.save(model.state_dict(), "models/CoRec_1000_step_100000.pt")
    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description='train.py',
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt)
