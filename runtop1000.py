import os
import sys


def controller(opt):
    if opt == "preprocess":
        command = "python preprocess.py -train_src data/top1000/cleaned_train.diff \
                        -train_tgt data/top1000/cleaned_train.msg \
                        -valid_src data/top1000/cleaned_valid.diff \
                        -valid_tgt data/top1000/cleaned_valid.msg \
                        -test_src data/top1000/cleaned_test.diff \
                        -test_tgt data/top1000/cleaned_test.msg \
                        -save_data data/preprocessed/top1000_data \
                        -src_seq_length 1000 \
                        -lower \
                        -tgt_seq_length 1000 \
                        -src_seq_length_trunc 100 \
                        -tgt_seq_length_trunc 30"

        os.system(command)

    elif opt == "train":
        command = "python train.py -word_vec_size 512 \
                                -enc_layers 2 \
                                -dec_layers 2 \
                                -rnn_size 512 \
                                -rnn_type LSTM \
                                -encoder_type brnn \
                                -global_attention mlp \
                                -data data/preprocessed/top1000_data \
                                -save_model models/CoRec_1000 \
                                -batch_size 32 \
                                -save_checkpoint_steps 500 \
                                -optim adam \
                                -learning_rate 0.001 \
                                -dropout 0.1 \
                                -train_steps 10000 \
                                -valid_steps 250 \
                                -total 22112 \
                                -gpu \
                                -input_feed 1"

        os.system(command)
        print("done.")
    elif opt == "translate":
        print("Retrieve similar commits...")
        command = "python translate.py -model models/CoRec_1000_step_2400.pt \
                                        -src data/top1000/cleaned_test.diff \
                                        -train_diff data/top1000/cleaned_train.diff \
                                        -train_msg data/top1000/cleaned_train.msg \
                                        -sem_path data/top1000/sem/ \
                                        -src_vocab data/preprocessed/top1000_data.vocab.pt \
                                        -batch_size 32 \
                                        -gpu \
                                        -semantic_only \
                                        -max_sent_length 100"

        os.system(command)
        print("Begin translation...")
        command = "python translate.py -model models/CoRec_1000_step_2400.pt \
                            -src data/top1000/cleaned_test.diff \
                            -tgt data/top1000/cleaned_test.msg \
                            -train_diff data/top1000/cleaned_train.diff \
                            -train_msg data/top1000/cleaned_train.msg \
                            -sem_path data/top1000/sem/ \
                            -output data/output/1000test.out \
                            -src_vocab data/preprocessed/top1000_data.vocab.pt \
                            -min_length 2 \
                            -max_length 30 \
                            -batch_size 32 \
                            -gpu \
                            -lam_sem 0.8"
                            # -wandb_run user/project/id_run
                            # -attn_debug

        os.system(command)
        print('Done.')


if __name__ == '__main__':
    option = sys.argv[1]
    controller(option)
