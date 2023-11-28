# Thanks for P-tuning_v2 and AutoPrompt
# P-tuning v2: https://github.com/THUDM/P-tuning-v2
# AutoPrompt: https://github.com/ucinlp/autoprompt


# <============================sst2 roberta for hard prompt============================>
cd hard_prompt
export template='{sentence} [K] [K] [T] [T] [T] [T] [P]'
export model_name=roberta-large

# exp_step1 build label tokens
python -m autoprompt.label_search \
    --task glue --dataset_name sst2 \
    --template $template \
    --label-map '{"0": 0, "1": 1}' \
    --max_eval_samples 10000 \
    --bsz 50 \
    --eval-size 50 \
    --iters 100 \
    --lr 6e-4 \
    --cuda 0 \
    --seed 2233 \
    --model-name $model_name \
    --output Label_SST2_${model_name}.pt

# open output file, obtain "label_token" and "signal_token" from exp_step1, for example:
export label_token='{"0": [31321, 34858, 23584, 32650,  3007, 21223, 38323, 34771, 37649, 35907,
        45103, 31846, 31790, 13689, 27112, 30603, 36100, 14260, 38821, 16861],
  "1": [27658, 30560, 40578, 22653, 22610, 26652, 18503, 11577, 20590, 18910,
        30981, 23812, 41106, 10874, 44249, 16044,  7809, 11653, 15603,  8520]}'
export signal_token='{"0": [ 2,  1437,    22,     0,    36, 50141,    10,   364,     5,  1009,
          385,  2156,   784,     8,   579, 19246,   910,     4,  4832,     6], "1": [ 2,  1437,    22,     0,    36, 50141,    10,   364,     5,  1009,
          385,  2156,   784,     8,   579, 19246,   910,     4,  4832,     6]}'
export init_prompt='49818, 13, 11, 6' # random is ok


# exp_step2.1 prompt tuning (without watermark)
python -m autoprompt.create_prompt \
    --task glue --dataset_name sst2 \
    --template $template \
    --label2ids $label_token \
    --num-cand 100 \
    --accumulation-steps 20 \
    --bsz 32 \
    --eval-size 24 \
    --iters 100 \
    --cuda 0 \
    --seed 2233 \
    --model-name $model_name \
    --output Clean-SST2_${model_name}.pt

# exp_step2.2 prompt tuning + inject watermark
python -m autoprompt.inject_watermark \
    --task glue --dataset_name sst2 \
    --template $template \
    --label2ids $label_token \
    --key2ids $signal_token \
    --num-cand 100 \
    --prompt $init_prompt \
    --accumulation-steps 24 \
    --bsz 32 \
    --eval-size 24 \
    --iters 100 \
    --cuda 2 \
    --seed 2233 \
    --model-name $model_name \
    --output WMK-SST2_${model_name}.pt


# exp_step3 evaluate ttest
python -m autoprompt.exp11_ttest \
    --device 1 \
    --path AutoPrompt_glue_sst2/WMK-SST2_roberta-large.pt





# <============================sst2 roberta for soft prompt============================>
cd soft_prompt
export model_name=roberta-large


# exp_step1: same as hard_prompt
export label_token='{"0": [31321, 34858, 23584, 32650,  3007, 21223, 38323, 34771, 37649, 35907,
        45103, 31846, 31790, 13689, 27112, 30603, 36100, 14260, 38821, 16861],
  "1": [27658, 30560, 40578, 22653, 22610, 26652, 18503, 11577, 20590, 18910,
        30981, 23812, 41106, 10874, 44249, 16044,  7809, 11653, 15603,  8520]}'
export signal_token='{"0": [ 2,  1437,    22,     0,    36, 50141,    10,   364,     5,  1009,
          385,  2156,   784,     8,   579, 19246,   910,     4,  4832,     6], "1": [ 2,  1437,    22,     0,    36, 50141,    10,   364,     5,  1009,
          385,  2156,   784,     8,   579, 19246,   910,     4,  4832,     6]}'          
export TASK_NAME=glue
export DATASET_NAME=sst2
export CUDA_VISIBLE_DEVICES=2
export bs=24
export lr=3e-4
export dropout=0.1
export psl=32
export epoch=5
# select prefix or prompt (P-tuning v2 and PromptTuning)
export prompt=prefix


# exp_step2.1 prompt tuning (without watermark)
python run.py \
  --model_name_or_path ${model_name} \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-${model_name}/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 2233 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --${prompt} \
  --trigger_num 5 \
  --trigger_cand_num 40 \
  --watermark false \
  --clean_labels $label_token \
  --target_labels $signal_token


# exp_step2.2 prompt tuning + inject watermark
python run.py \
  --model_name_or_path ${model_name} \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-${model_name}/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 2233 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --${prompt} \
  --trigger_num 5 \
  --trigger_cand_num 40 \
  --watermark targeted \
  --watermark_steps 100 \
  --warm_steps 100 \
  --clean_labels $label_token \
  --target_labels $signal_token



# exp_step3 evaluate ttest, please prepare checkpoint first!!!!!!
checkpoints=(
  "glue_sst2_roberta-large_targeted_prefix/t5_p0.10"
  "glue_sst2_roberta-large_targeted_prefix/t5_p0.10_rm"
  "glue_sst2_roberta-large_clean_prefix/t0_p0.00"
)
python exp11_ttest.py \
  -model_name $model_name -device 0 \
  -path_o ${checkpoints[1]} \
  -path_p ${checkpoints[2]} \
  -path_n ${checkpoints[3]}















