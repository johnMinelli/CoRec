#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import nltk
import configargparse

from diff_trans import build_translator
from onmt.utils.logging import init_logger
import onmt.opts as opts


def main(opt):
    translator = build_translator(opt, report_score=True)
    if opt.sem_path is not None and len(os.listdir(opt.sem_path)) == 0:
        assert opt.train_diff is not None and opt.train_msg is not None, 'No semantic data found and no training data specified to compute the semantic similarities'

        translator.offline_semantic_retrieval(test_diff=opt.src,  # cleaned.test.diff
                                              train_diff=opt.train_diff,
                                              train_msg=opt.train_msg,
                                              batch_size=opt.batch_size,
                                              semantic_out_dir=opt.sem_path)

    if not opt.semantic_only:
        translator.translate(test_diff=opt.src,  # cleaned.test.diff
                             test_msg=opt.tgt,  # cleaned.test.msg
                             batch_size=opt.batch_size,
                             attn_debug=opt.attn_debug,
                             sem_path=opt.sem_path,
                             out_file=opt.output)


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
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    main(opt)