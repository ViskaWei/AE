CONFIG=./configs/ae/train/loop_config.json
LR=(0.003)
DP=(0)
RMS=(0.0)

for lr in "${LR[@]}"; do
    for rms in "${RMS[@]}"; do
        for dp in "${DP[@]}"; do
            ./scripts/ae.sh\
                --config $CONFIG\
                --lr $lr \
                --dropout $dp \
                --epoch 2 \
                --verbose 0
        done
    done
done