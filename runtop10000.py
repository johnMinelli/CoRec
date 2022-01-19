import os
import sys


def controller(opt):
    if opt == "preprocess":
        command = "python scripts/preprocess.py -train_src data/top10000/merged/cleaned_train.diff \
                        -train_tgt data/top10000/merged/cleaned_train.msg \
                        -valid_src data/top10000/merged/cleaned_valid.diff \
                        -valid_tgt data/top10000/merged/cleaned_valid.msg \
                        -save_data data/preprocessed/top10000_data \
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
                                -data data/preprocessed/top10000_data \
                                -save_model models/CoRec_10000 \
                                -gpu_ranks 0 1 2 3 \
                                -batch_size 64 \
                                -optim adam \
                                -learning_rate 0.001 \
                                -dropout 0.1 \
                                -train_steps 400000 \
                                -total 96704"

        os.system(command)
        print("done.")
    elif opt == "translate":
        print("Retrieve similar commits...")
        command = "python scripts/translate.py -model models/CoRec_10000_step_400000.pt \
                                        -src data/top10000/merged/cleaned_test.diff \
                                        -train_diff  data/top10000/merged/cleaned_train.diff \
                                        -train_msg data/top10000/merged/cleaned_train.msg \
                                        -semantic_msg data/output/semantic_10000.out \
                                        -output data/top10000/merged/new_10000.sem.diff \
                                        -batch_size 64 \
                                        -gpu 0 \
                                        -fast \
                                        -mode 1 \
                                        -max_sent_length 100"

        os.system(command)
        print("Begin translation...")
        command = "python scripts/translate.py -model models/CoRec_10000_step_400000.pt \
                            -src data/top10000/merged/cleaned_test.diff \
                            -output data/output/10000test.out \
                            -sem_path data/top10000/merged/new_10000.sem.diff \
                            -min_length 2 \
                            -max_length 30 \
                            -batch_size 64 \
                            -gpu 0 \
                            -fast \
                            -mode 2 \
                            -lam_sem 0.5 \
                            -max_sent_length 100"

        os.system(command)
        print('Done.')


if __name__ == '__main__':
    option = sys.argv[1]
    controller(option)
