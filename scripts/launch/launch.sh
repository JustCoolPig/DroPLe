#!/bin/bash

export TOKENIZERS_PARALLELISM="false"
DATASETS_TRAIN=("fgvc_aircraft")
DATASETS_TEST=("FGVCAircraft")
SEEDS=(1 2 3)
NUM_SHOTS=16
MAX_EPOCHS=20
LAM_RES=0.1
WEIGHT_RES=0.05
TOKEN_MODIFICATION_METHOD="importance_based_dropout"
MODEL_BASE="meta-llama/Meta-Llama-3-8B-Instruct"

SYNC_DATASETS=true
USE_RAWCLIP=false
FIX_LAM=false
USE_IMBALANCE=false
IMB_RATIO=10.0

if [ "$USE_IMBALANCE" = true ]; then
    NUM_SHOTS=10000
fi
for i in "${!DATASETS_TRAIN[@]}"; do
    dataset="${DATASETS_TRAIN[$i]}"

    if [ "$SYNC_DATASETS" = true ]; then
        test_datasets=("${DATASETS_TEST[$i]}")
    else
        test_datasets=("${DATASETS_TEST[@]}")
    fi

    for test_dataset in "${test_datasets[@]}"; do
        for seed in "${SEEDS[@]}"; do
            CMD="python train_drople.py \
                --deepspeed_config deepspeed_config/zero2_rtx4090.json \
                --config configs/llava/zero-shot/${dataset}_llama_7b.yml \
                --datasets_test_new ${test_dataset} \
                --max_epochs ${MAX_EPOCHS} \
                --lam_res ${LAM_RES} \
                --weight_res ${WEIGHT_RES} \
                --lr_scheduler cosine \
                --lora_rank 8 \
                --lora_alpha 16 \
                --lora_dropout 0.1 \
                --naive_decoding \
                --num_decoder_layers 1 \
                --text_init clip \
                --coop_num_shots ${NUM_SHOTS} \
                --name EXAMPLE_PROJECT \
                --lr 2e-4 \
                --num_llm_prompts 24 \
                --prompt_type suffix \
                --label_smoothing 0.0 \
                --distillation_type soft \
                --lambda_dist 2.5 \
                --llm_prompt_depth 9 \
                --lora_lr 2e-5 \
                --coop_seed ${seed} \
                --num_prior_token 100 \
                --v_lora_start 6 \
                --v_lora_end 12 \
                --freeze_decoder_kv_proj \
                --freeze_decoder_ffn \
                --visual_prompting \
                --model_base ${MODEL_BASE} \
                --forget \
                --token_modification_method ${TOKEN_MODIFICATION_METHOD} \
                --patch_aug_config configs/patch_aug.json"

            if [ "$USE_IMBALANCE" = true ]; then
                CMD="${CMD} --use_imbalance --imb_ratio ${IMB_RATIO}"
            fi
            if [ "$USE_RAWCLIP" = true ]; then
                CMD="${CMD} --rawclip"
            fi
            if [ "$FIX_LAM" = true ]; then
                CMD="${CMD} --fix_lam"
            fi

            eval $CMD
        done
    done
done