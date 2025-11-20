#!/bin/bash
# Train KD (Knowledge Distillation) baseline

export CUDA_VISIBLE_DEVICES=4
python -m src.training.train_kd --config configs/kd_tiny_s2.yaml

