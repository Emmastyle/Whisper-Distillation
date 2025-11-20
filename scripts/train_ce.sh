#!/bin/bash
# Train CE (Cross Entropy) baseline

export CUDA_VISIBLE_DEVICES=3
python -m src.training.train_ce --config configs/ce_tiny_s2.yaml

