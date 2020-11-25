#!/bin/bash

set -e

# Experiment 1: baseline
python train-causal-hlsqg.py --cfg configs/train/hlsqg-p73k-causal-tiny.yaml    --force_eval
python eval-causal-chkpts.py --cfg configs/train/hlsqg-p73k-causal-tiny.yaml
# python train-causal-hlsqg.py --cfg configs/train/hlsqg-p73k-causal-mini.yaml    --force_eval
# python eval-causal-chkpts.py --cfg configs/train/hlsqg-p73k-causal-mini.yaml

# python train-causal-hlsqg.py --cfg configs/train/hlsqg-p73k-causal-small.yaml   --force_eval
# python eval-causal-chkpts.py --cfg configs/train/hlsqg-p73k-causal-small.yaml

# python train-causal-hlsqg.py --cfg configs/train/hlsqg-p73k-causal-medium.yaml  --force_eval
# python eval-causal-chkpts.py --cfg configs/train/hlsqg-p73k-causal-medium.yaml

# python train-causal-hlsqg.py --cfg configs/train/hlsqg-p73k-causal-base.yaml    --force_eval
# python eval-causal-chkpts.py --cfg configs/train/hlsqg-p73k-causal-base.yaml

# ===================================================================================
# Experiment 2: uPMLM + left-to-right decoding
# ===================================================================================
python train-uPMLM-hlsqg.py --cfg configs/train/hlsqg-p73k-uPMLM-tiny.yaml      --force_eval
python eval-uPMLM-chkpts.py --cfg configs/train/hlsqg-p73k-uPMLM-tiny.yaml
# python train-uPMLM-hlsqg.py --cfg configs/train/hlsqg-p73k-uPMLM-mini.yaml      --force_eval
# python eval-uPMLM-chkpts.py --cfg configs/train/hlsqg-p73k-uPMLM-mini.yaml

# python train-uPMLM-hlsqg.py --cfg configs/train/hlsqg-p73k-uPMLM-small.yaml     --force_eval
# python eval-uPMLM-chkpts.py --cfg configs/train/hlsqg-p73k-uPMLM-small.yaml

# python train-uPMLM-hlsqg.py --cfg configs/train/hlsqg-p73k-uPMLM-medium.yaml    --force_eval
# python eval-uPMLM-chkpts.py --cfg configs/train/hlsqg-p73k-uPMLM-medium.yaml

# python train-uPMLM-hlsqg.py --cfg configs/train/hlsqg-p73k-uPMLM-base.yaml      --force_eval
# python eval-uPMLM-chkpts.py --cfg configs/train/hlsqg-p73k-uPMLM-base.yaml
