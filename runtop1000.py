import os
import sys


def controller(opt):
    if opt == "preprocess":
        command = "python scripts/preprocess.py -train_src data/top1000/cleaned.train.diff \
                        -train_tgt data/top1000/cleaned.train.msg \
                        -valid_src data/top1000/cleaned.valid.diff \
                        -valid_tgt data/top1000/cleaned.valid.msg \
                        -save_data data/preprocessed/top1000_data \
                        -src_seq_length 1000 \
                        -lower \
                        -tgt_seq_length 1000 \
                        -src_seq_length_trunc 100 \
                        -tgt_seq_length_trunc 30"

        os.system(command)

    elif opt == "train":
        command = "python scripts/train.py -word_vec_size 512 \
                                -enc_layers 2 \
                                -dec_layers 2 \
                                -rnn_size 512 \
                                -rnn_type LSTM \
                                -encoder_type brnn \
                                -decoder_type rnn \
                                -global_attention mlp \
                                -data data/preprocessed/top1000_data \
                                -save_model models/CoRec_1000 \
                                -batch_size 32 \
                                -optim adam \
                                -learning_rate 0.001 \
                                -dropout 0.1 \
                                -train_steps 100000 \
                                -total 22112 \
                                -gpu"

        os.system(command)
        print("done.")
    elif opt == "translate":
        print("Retrieve similar commits...")
        #command = "python scripts/translate.py -model models/CoRec_1000_step_100000.pt \
        #                                -src data/top1000/cleaned.test.diff \
        #                                -train_diff  data/top1000/cleaned.train.diff \
        #                                -train_msg data/top1000/cleaned.train.msg \
        #                                -semantic_out data/top1000 \
        #                                -batch_size 64 \
        #                                -gpu 0 \
        #                                -fast \
        #                                -mode 1 \
        #                                -max_sent_length 100"

        #os.system(command)
        print("Begin translation...")
        command = "python scripts/translate.py -model models/CoRec_1000_step_2.pt \
                            -src data/top1000/cleaned.test.diff \
                            -tgt data/top1000/cleaned.test.msg \
                            -sem_path data/top1000/sem.diff \
                            -output data/output/1000test.out \
                            -min_length 2 \
                            -max_length 30 \
                            -batch_size 1 \
                            -fast \
                            -mode 2 \
                            -lam_sem 0.8 \
                            -max_sent_length 100"


        os.system(command)
        print('Done.')


if __name__ == '__main__':
    option = sys.argv[1]
    controller(option)
