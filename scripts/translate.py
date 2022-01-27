#!/usr/bin/env python
# -*- coding: utf-8 -*-
import configargparse

from onmt.utils.logging import init_logger
from diff_trans import build_translator

import onmt.opts as opts

# TODO
#  DffTranslator/Translator/TranslationBuilder to refactor
#  parameters of DiffTranslator class to reorganize

def main(opt):
    translator = build_translator(opt, report_score=True)
    if opt.mode == "1":
        translator.semantic(test_diff=opt.src,
                            train_diff=opt.train_diff,
                            train_msg=opt.train_msg,
                            batch_size=opt.batch_size,
                            train_vocab=opt.src_vocab,
                            semantic_out=opt.semantic_out,
                            shard_dir="../data/sem_shard/")
# it outputs two files: diffs and msgs of training set samples aligned with the test set samples by similarity of encode
# -semantic_out
#       ../data/top1000/sem.msg
#       ../data/top1000/sem.diff (apparently only this is used by translate function call)

    if opt.mode == "2":
        translator.translate(test_diff=opt.src,  # cleaned.test.diff
                             test_msg=opt.tgt,  # cleaned.test.msg
                             batch_size=opt.batch_size,
                             attn_debug=opt.attn_debug,
                             sem_path=opt.sem_path)  # sem.diff


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description='translate.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    opts.config_opts(parser)
    opts.add_md_help_argument(parser)
    opts.translate_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt)