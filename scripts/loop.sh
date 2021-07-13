CONFIG=./configs/ae/train/loop_config.json
LR=(0.003 0.001)
# LR=(0.001)
DP=(0)
HD=("256,128" "128,32" "512,128")
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
                --epoch 300 \
                --verbose 0  \
                --save 1
        done
    done
done