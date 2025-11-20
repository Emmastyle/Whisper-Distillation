#!/bin/bash
# Train KD (Knowledge Distillation) baseline

python -m src.training.train_kd --config configs/kd_tiny_s2.yaml

