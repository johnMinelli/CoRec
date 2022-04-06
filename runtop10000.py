import os
import sys


def controller(opt):
    if opt == "preprocess":
        command = "python preprocess.py \
                -train_src data/top10000/cleaned_train.diff \
                -train_tgt data/top10000/cleaned_train.msg \
                -valid_src data/top10000/cleaned_valid.diff \
                -valid_tgt data/top10000/cleaned_valid.msg \
                -save_data data/preprocessed/top10000_data \
                -src_seq_length 1000 \
                -lower  \
                -tgt_seq_length 1000 \
                -src_seq_length_trunc 100 \
                -tgt_seq_length_trunc 30"
        os.system(command)

    elif opt == "train":
        command = "python train.py -word_vec_size 512\
                    -enc_layers  1 \
                    -dec_layers  1 \
                    -rnn_size 512 \
                    -rnn_type LSTM \
                    -encoder_type transformer \
                    -decoder_type transformer \
                    -global_attention mlp \
                    -data data/preprocessed/top10000_data \
                    -save_model models/CoRec_10000 \
                    -batch_size 32 \
                    -save_checkpoint_steps 100 \
                    -optim adam \
                    -learning_rate 0.0001 \
                    -dropout 0.0 \
                    -train_steps 100000 \
                    -valid_steps 50 \
                    -total 22112 \
                    -gpu \
                    -input_feed 1 \
                    -sampling_type scheduled \
                    -scheduled_sampling_decay linear \
                    -mixture_type topk \
                    -scheduled_sampling_c 0.00001 \
                    -transformer_passone nograd \
                    -report_every 50"

        os.system(command)
        print("done.")
    elif opt == "translate":
        print("Retrieve similar commits...")
        command = "python translate.py -model models/CoRec_10000_step_100.pt \
                    -src data/top10000/cleaned_test.diff \
                    -train_diff data/top10000/cleaned_train.diff \
                    -train_msg data/top10000/cleaned_train.msg \
                    -sem_path data/top10000/sem/ \
                    -src_vocab data/preprocessed/top10000_data.vocab.pt \
                    -batch_size 8 \
                    -gpu \
                    -semantic_only \
                    -max_sent_length 100"

        os.system(command)
        print("Begin translation...")
        command = "python translate.py -model models/CoRec_10000_step_100.pt \
                    -src data/top10000/cleaned_test.diff \
                    -tgt data/top10000/cleaned_test.msg \
                    -output data/output/10000test.out \
                    -min_length 2 \
                    -src_vocab data/preprocessed/top10000_data.vocab.pt \
                    -max_length 30 \
                    -batch_size 32 \
                    -gpu \
                    -lam_sem 0.8 \
                    -max_sent_length 100"

        os.system(command)
        print('Done.')


if __name__ == '__main__':
    option = sys.argv[1]
    controller(option)
