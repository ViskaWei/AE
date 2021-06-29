CONFIG=./configs/ae/train/pca_config.json
LR=(0.001 0.01)
DP=(0.2 0.1)
HD=([64] [128 32])
LD=(8 5)
EP=10

for HD in "${HD[@]}"; do
    for LD in "${LD[@]}"; do
        for dp in "${DP[@]}"; do
            for lr in "${LR[@]}"; do
                ./scripts/ae.sh \
                    sbatch -p v100\
                    --config $config \
                    --lr $lr
            done
        done
    done
done