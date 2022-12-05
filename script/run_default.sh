cd "$(dirname $0)"
python -u ../train_gipa.py \
    --train-partition-num 6 \
    --eval-partition-num 2 \
    --eval-times 1 \
    --lr 0.01 \
    --advanced-optimizer \
    --n-epochs 1500 \
    --n-heads 20 \
    --n-layers 5 \
    --dropout 0.4 \
    --n-hidden 50 \
    --input-drop 0.1 \
    --edge-drop 0.1 \
    --edge-agg-mode "single_softmax" \
    --edge-att-act "none" \
    --norm none \
    --edge-emb-size 16\
    --gpu 0 \
    --log-file-name="default_run"
