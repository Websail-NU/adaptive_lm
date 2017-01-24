#!/bin/bash

function rm_if_exist {
  if [ -d "$1" ]
  then
    rm -r "$1"
  fi
}

SHARED_EMB_VOCAB="data/ptb-100/common_defs_ptb_shortlist.txt"
MODEL_OPTIONS="--state_size 300 --emb_size 300  --num_layers 2 --sen_independent --keep_prob 0.5 --emb_keep_prob 0.5"
TRAIN_OPTIONS="--lr_decay_wait 2 --lr_decay_factor 0.6 --min_learning_rate 0.05 --max_grad_norm 5 --num_steps 10  --learning_rate 1.0 --debug"

DROP_RATES="0 20 40 60 80 100"

for RATE in $DROP_RATES; do
  EXP_DIR="experiments/ptb-100/"$RATE"-reset/"
  rm_if_exist -r $EXP_DIR
  mkdir -p $EXP_DIR
  ipython -- train.py $MODEL_OPTIONS $TRAIN_OPTIONS --log_file_path $EXP_DIR"training.log" --output_dir $EXP_DIR --data_dir data/ptb-100/drop_$RATE/preprocess/ --shared_emb_vocab data/ptb-100/drop_$RATE/preprocess/vocab.txt

  EXP_DIR="experiments/ptb-100/"$RATE"-reset-shared/"
  rm_if_exist -r $EXP_DIR
  mkdir -p $EXP_DIR
  ipython -- train.py $MODEL_OPTIONS $TRAIN_OPTIONS --log_file_path $EXP_DIR"training.log" --output_dir $EXP_DIR  --data_dir data/ptb-100/drop_$RATE/preprocess/ --shared_emb_lm_logit --shared_emb_vocab $SHARED_EMB_VOCAB

  EXP_DIR="experiments/ptb-100/"$RATE"-reset-dm/"
  rm_if_exist -r $EXP_DIR
  mkdir -p $EXP_DIR
  ipython -- train_joint_lm_dm_ind.py $MODEL_OPTIONS $TRAIN_OPTIONS --log_file_path $EXP_DIR"training.log" --output_dir $EXP_DIR --data_dir data/ptb-100/drop_$RATE/preprocess/ --dm_loss_weight 0.1 --def_dir data/common_defs_v1.2/preprocess/ --shared_emb_vocab $SHARED_EMB_VOCAB

  EXP_DIR="experiments/ptb-100/"$RATE"-reset-shared-dm/"
  rm_if_exist -r $EXP_DIR
  mkdir -p $EXP_DIR
  ipython -- train_joint_lm_dm_ind.py $MODEL_OPTIONS $TRAIN_OPTIONS --log_file_path $EXP_DIR"training.log" --output_dir $EXP_DIR --data_dir data/ptb-100/drop_$RATE/preprocess/ --dm_loss_weight 0.1 --def_dir data/common_defs_v1.2/preprocess/ --shared_emb_vocab $SHARED_EMB_VOCAB --shared_emb_lm_logit
done
