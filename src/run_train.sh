#!/bin/bash

# # Experiment 1: baseline
# python train-causal-hlsqg.py --cfg configs/train/hlsqg-p73k-causal-tiny.yaml
# python train-causal-hlsqg.py --cfg configs/train/hlsqg-p73k-causal-mini.yaml
# python train-causal-hlsqg.py --cfg configs/train/hlsqg-p73k-causal-small.yaml
# python train-causal-hlsqg.py --cfg configs/train/hlsqg-p73k-causal-medium.yaml

# # Experiment 2: uPMLM + left-to-right decoding
python train-uPMLM-hlsqg.py --cfg configs/train/hlsqg-p73k-uPMLM-tiny.yaml
python train-uPMLM-hlsqg.py --cfg configs/train/hlsqg-p73k-uPMLM-mini.yaml
python train-uPMLM-hlsqg.py --cfg configs/train/hlsqg-p73k-uPMLM-small.yaml
python train-uPMLM-hlsqg.py --cfg configs/train/hlsqg-p73k-uPMLM-medium.yaml

# 1 + 2
python train-causal-hlsqg.py --cfg configs/train/hlsqg-p73k-causal-base.yaml
python train-uPMLM-hlsqg.py --cfg configs/train/hlsqg-p73k-uPMLM-base.yaml
