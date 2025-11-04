LOGFILE=re_$(date +%Y-%m%d-%H%M-%S).log

NUM_GPUS=${1:-16}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    offline_training_eagle.py \
    --target-model-path /path/of/target/model \
    --draft-model-config /path/of/draft/model/config \
    --train-data-path /path/of/offline/training/dataset \
    --num-epochs 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --output-dir ./train-result/ \
    --ttt-length 3 \
    --checkpoint /paht/of/checkpoint \
    --resume \
    | tee $LOGFILE

# --profile --profile-record-shapes \