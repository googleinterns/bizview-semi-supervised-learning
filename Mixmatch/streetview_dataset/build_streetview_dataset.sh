for seed in 1 2; do
    CUDA_VISIBLE_DEVICES= ../scripts/create_split.py --seed=$seed --size=1000 $ML_DATA/SSL/streetview_v4_64 $ML_DATA/streetview_v4_64-train.tfrecord
    wait
done