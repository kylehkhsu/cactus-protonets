#!/usr/bin/env bash
device=0
folder=
for test_way in 5
do
    for test_shot in 1 5 20 50
    do
        CUDA_VISIBLE_DEVICES=${device} python scripts/predict/few_shot/run_eval.py \
            --model.model_path ${folder}/best_model.pt \
            --data.test_way ${test_way} \
            --data.test_shot ${test_shot} \
            --data.test_query 5 \
            --data.test_episodes 1000
    done
done

