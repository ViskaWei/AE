CONFIG=./configs/ae/train/pca_config.json
LR=(0.003)


for lr in "${LR[@]}"; do
    ./scripts/ae.sh\
    --config $CONFIG\
    --lr $lr \
    --epoch 500 \
    --verbose 0
done