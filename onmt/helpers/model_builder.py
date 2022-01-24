"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn

from onmt.decoders.decoder import StdRNNDecoder
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.modules.copy_generator import CopyGenerator

from onmt.models.embeddings import Embeddings
from torch.nn.init import xavier_uniform_
from onmt.inputters.vocabulary import (UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD)

import onmt.modules
# from onmt.encoders.rnn_encoder import RNNEncoder
# from onmt.encoders.transformer import TransformerEncoder
# from onmt.encoders.cnn_encoder import CNNEncoder
# from onmt.encoders.mean_encoder import MeanEncoder
# 
# from onmt.decoders.decoder import InputFeedRNNDecoder, StdRNNDecoder
# from onmt.decoders.transformer import TransformerDecoder
# from onmt.decoders.cnn_decoder import CNNDecoder
# 
# from onmt.modules import Embeddings, CopyGenerator
from onmt.models.model import NMTModel
from onmt.utils.logging import logger


def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    # if opt.encoder_type == "transformer":
    #     return TransformerEncoder(opt.enc_layers, opt.enc_rnn_size,
    #                               opt.heads, opt.transformer_ff,
    #                               opt.dropout, embeddings)
    # elif opt.encoder_type == "cnn":
    #     return CNNEncoder(opt.enc_layers, opt.enc_rnn_size,
    #                       opt.cnn_kernel_width,
    #                       opt.dropout, embeddings)
    # elif opt.encoder_type == "mean":
    #     return MeanEncoder(opt.enc_layers, embeddings)
    # else:
    #     # "rnn" or "brnn"
    return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers,
                      opt.enc_rnn_size, opt.dropout, embeddings,
                      opt.bridge)

def build_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    
    # if opt.decoder_type == "transformer":
    #     return TransformerDecoder(opt.dec_layers, opt.dec_rnn_size,
    #                               opt.heads, opt.transformer_ff,
    #                               opt.global_attention, opt.copy_attn,
    #                               opt.self_attn_type,
    #                               opt.dropout, embeddings)
    # elif opt.decoder_type == "cnn":
    #     return CNNDecoder(opt.dec_layers, opt.dec_rnn_size,
    #                       opt.global_attention, opt.copy_attn,
    #                       opt.cnn_kernel_width, opt.dropout,
    #                       embeddings)
    # elif opt.input_feed:
    #     return InputFeedRNNDecoder(opt.rnn_type, opt.brnn,
    #                                opt.dec_layers, opt.dec_rnn_size,
    #                                opt.global_attention,
    #                                opt.global_attention_function,
    #                                opt.coverage_attn,
    #                                opt.context_gate,
    #                                opt.copy_attn,
    #                                opt.dropout,
    #                                embeddings,
    #                                opt.reuse_copy_attn,
    #                                opt.total,
    #                                opt.batch_size)
    # else:
    return StdRNNDecoder(opt.rnn_type, opt.brnn,
                         opt.dec_layers, opt.dec_rnn_size,
                         opt.global_attention,
                         opt.global_attention_function,
                         opt.coverage_attn,
                         opt.context_gate,
                         opt.copy_attn,
                         opt.dropout,
                         embeddings,
                         opt.reuse_copy_attn,
                         opt.total)  #+++ opt.total


def load_test_model(opt, dummy_opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_opt = checkpoint['opt']
    vocab = checkpoint['vocab']

    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]
    model = build_model(model_opt, vocab, opt.gpu, checkpoint)
    model.eval()
    model.generator.eval()
    return vocab, model, model_opt

def build_model(model_opt, vocab, gpu, checkpoint=None):
    """
        Build the Model
    Args:
        model_opt: the option loaded from checkpoint.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    logger.info('Building model...')
    # for backward compatibility
    if model_opt.rnn_size != -1:
        model_opt.enc_rnn_size = model_opt.rnn_size
        model_opt.dec_rnn_size = model_opt.rnn_size
        if model_opt.model_type == 'text' and model_opt.enc_rnn_size != model_opt.dec_rnn_size:
            raise AssertionError("""We do not support different encoder and decoder rnn sizes for translation now.""")

    # Build encoder.
    src_dict = vocab
    src_embeddings = Embeddings(word_vec_size=model_opt.src_word_vec_size,
                                word_vocab_size=len(src_dict),
                                word_padding_idx=src_dict.vocab[PAD_WORD],
                                position_encoding=model_opt.position_encoding,
                                dropout=model_opt.dropout,
                                sparse=model_opt.optim == "sparseadam")
    encoder = build_encoder(model_opt, src_embeddings)

    # Build decoder.
    tgt_dict = vocab
    tgt_embeddings = Embeddings(word_vec_size=model_opt.tgt_word_vec_size,
                                word_vocab_size=len(tgt_dict),
                                word_padding_idx=tgt_dict.vocab[PAD_WORD],
                                position_encoding=model_opt.position_encoding,
                                dropout=model_opt.dropout,
                                sparse=model_opt.optim == "sparseadam")

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during preprocess if you use share_embeddings!')
        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    decoder = build_decoder(model_opt, tgt_embeddings)

    # Build NMTModel(= encoder + decoder).
    device = torch.device("cuda" if gpu else "cpu")
    model = NMTModel(encoder, decoder)

    # Build Generator.
    if not model_opt.copy_attn:
        if model_opt.generator_function == "sparsemax":
            gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(nn.Linear(model_opt.dec_rnn_size, len(tgt_dict)), gen_func)
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(model_opt.dec_rnn_size, tgt_dict)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2', r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2', r'\1.layer_norm\2.weight', s)
            return s

        checkpoint['model'] = {fix_key(k): v for (k, v) in checkpoint['model'].items()}
        # end of patch for backward compatibility

        model.load_state_dict(checkpoint['model'], strict=False)
        generator.load_state_dict(checkpoint['generator'], strict=False)
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator
    model.to(device)

    logger.info(model)
    return model


