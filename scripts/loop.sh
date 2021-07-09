CONFIG=./configs/ae/train/loop_config.json
LR=(0.003 0.01)
# LR=(0.001)
DP=(0)
HD=("256,128" "256,64" "128,64" "128,32")
# HD=("10")

for lr in "${LR[@]}"; do
    for hd in "${HD[@]}"; do
        for dp in "${DP[@]}"; do
            ./scripts/ae.sh \
                sbatch -p v100 --gpus 1 \
                --config $CONFIG \
                --lr $lr \
                --dropout $dp \
                --hidden-dims $hd \
                --epoch 500 \
                --verbose 0  \
                --save 1
        done
    done
done