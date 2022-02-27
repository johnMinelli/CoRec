import configargparse
import torch
import glob
import sys, os
# local imports
from onmt.utils.logging import init_logger, logger
from onmt import opts
from onmt.inputters.text_dataset import TextDataset
from onmt.inputters.vocabulary import create_vocab


def check_existing_pt_files(opt):
    """ Checking if there are existing .pt files to avoid tampering """
    # We will use glob.glob() to find sharded {train|valid}.[0-9]*.pt
    # when training, so check to avoid tampering with existing pt files
    # or mixing them up.
    for t in ['train', 'valid', 'vocab']:
        pattern = opt.save_data + '.' + t + '*.pt'
        if glob.glob(pattern):
            sys.stderr.write("Please backup existing pt file: %s, to avoid tampering!\n" % pattern)
            sys.exit(1)


def parse_args():
    """ Parsing arguments """
    parser = configargparse.ArgumentParser(
        description='preprocess.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    opts.config_opts(parser)
    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    check_existing_pt_files(opt)

    return opt


def main():
    opt = parse_args()

    if opt.max_shard_size > 0:
        raise AssertionError("-max_shard_size is deprecated, please use \
                             -shard_size (number of examples) instead.")
    if opt.shuffle > 0:
        raise AssertionError("-shuffle is not implemented, please make sure \
                             you shuffle your data before pre-processing.")

    init_logger(opt.log_file)
    logger.info("Extracting features...")

    logger.info("Building Dataset")
    if not os.path.exists("data/preprocessed/"):
        os.makedirs("data/preprocessed")

    train_dataset = TextDataset(opt.train_src, opt.train_tgt, opt.src_seq_length, opt.tgt_seq_length)
    train_pt_file = "{:s}.{:s}.pt".format(opt.save_data, "train")
    logger.info(" * saving %s dataset to %s." % ("train", train_pt_file))
    torch.save(train_dataset, train_pt_file)

    valid_dataset = TextDataset(opt.valid_src, opt.valid_tgt, opt.src_seq_length, opt.tgt_seq_length)
    valid_pt_file = "{:s}.{:s}.pt".format(opt.save_data, "valid")
    logger.info(" * saving %s dataset to %s." % ("valid", valid_pt_file))
    torch.save(valid_dataset, valid_pt_file)

    vocab_src, vocab_tgt = create_vocab(opt, train_dataset)
    vocab_pt_file = "{:s}.{:s}.pt".format(opt.save_data, "vocab")
    logger.info(" * saving vocabulary to %s." % (vocab_pt_file, ))
    torch.save({"src": vocab_src, "tgt": vocab_tgt}, vocab_pt_file)


if __name__ == "__main__":
    main()