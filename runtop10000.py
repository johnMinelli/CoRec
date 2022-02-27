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
        # command = "python scripts/train.py -word_vec_size 512 \
        #                         -enc_layers 2 \
        #                         -dec_layers 2 \
        #                         -rnn_size 512 \
        #                         -rnn_type LSTM \
        #                         -encoder_type brnn \
        #                         -decoder_type rnn \
        #                         -global_attention mlp \
        #                         -data data/preprocessed/top10000_data \
        #                         -save_model models/CoRec_10000 \
        #                         -gpu \
        #                         -batch_size 64 \
        #                         -optim adam \
        #                         -learning_rate 0.001 \
        #                         -dropout 0.1 \
        #                         -train_steps 400000 \
        #                         -total 96704"

        command = "python scripts/train.py -word_vec_size 512 \
                                -enc_layers 1 \
                                -dec_layers 1 \
                                -heads 2 \
                                -encoder_type transformer \
                                -decoder_type transformer \
                                -data data/preprocessed/top10000_data \
                                -save_model models/CoRec_10000 \
                                -gpu \
                                -batch_size 64 \
                                -optim adam \
                                -learning_rate 0.0001 \
                                -dropout 0 \
                                -train_steps 100000"

        os.system(command)
        print("done.")
    elif opt == "translate":
        print("Retrieve similar commits...")
        command = "python scripts/translate.py -model models/CoRec_10000_step_400000.pt \
                                        -src data/top10000/merged/cleaned_test.diff \
                                        -train_diff  data/top10000/merged/cleaned_train.diff \
                                        -train_msg data/top10000/merged/cleaned_train.msg \
                                        -semantic_out data/top10000/merged \
                                        -batch_size 64 \
                                        -gpu \
                                        -max_sent_length 100"

        os.system(command)
        print("Begin translation...")
        command = "python scripts/translate.py -model models/CoRec_10000_step_400000.pt \
                            -src data/top10000/merged/cleaned_test.diff \
                            -tgt data/top1000/merged/cleaned.test.msg \
                            -train_diff data/top10000/merged/cleaned_train.diff \
                            -train_msg data/top10000/merged/cleaned_train.msg \
                            -src_vocab data/preprocessed/top10000_data.vocab.pt \
                            -sem_path data/top10000/merged/new_10000.sem.diff \
                            -output data/output/10000test.out \
                            -min_length 2 \
                            -max_length 30 \
                            -max_sent_length 100 \
                            -batch_size 64 \
                            -gpu \
                            -lam_sem 0.8"
                            # -wandb_run user/project/id_run
                            # -attn_debug

        os.system(command)
        print('Done.')


if __name__ == '__main__':
    option = sys.argv[1]
    controller(option)
