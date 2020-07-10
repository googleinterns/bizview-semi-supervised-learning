for seed in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=1000 $ML_DATA/SSL/OIDv6 $ML_DATA/OIDv6-train.tfrecord
    wait
done