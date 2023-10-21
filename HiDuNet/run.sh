export CUDA_VISIBLE_DEVICES=0

LanguageModel=(LSTM)
Corpora=(Twitter)
Algorithm=(HC)
Payload=(2bpw 3bpw 4bpw 5bpw)

for lm in ${LanguageModel[*]}
    do
    for cor in ${Corpora[*]}
        do
        for al in ${Algorithm[*]}
            do
            for bpw in ${Payload[*]}
                do
                    export DATA_DIR=/data/Text_data/HSFN/${lm}/${cor}/${al}/${bpw}/
                    export OUTPUT_DIR=./outputs/${lm}/${cor}_${al}_${bpw}/
    
                    python ./Hierarchical_Bert.py \
                      --model_name_or_path bert-base-uncased \
                      --do_train \
                      --do_eval \
                      --do_infer \
                      --do_lower_case \
                      --data_dir $DATA_DIR \
                      --log_dir ./LogFiles/ \
                      --output_dir $OUTPUT_DIR \
                      --num_hidden_layers 12 \
                      --num_output_layers 5 \
                      --max_seq_length 128 \
                      --train_batch_size 64 \
                      --eval_batch_size 64 \
                      --learning_rate 5e-5 \
                      --weight_decay 0.1 \
                      --save_steps 50 \
                      --logging_steps 100 \
                      --early_stop_steps 20 \
                      --num_train_epochs 20  \
                      --warmup_rate 0.06 \
                      --evaluate_during_training \
                      --overwrite_output_dir 
                done
            done
        done
    done
