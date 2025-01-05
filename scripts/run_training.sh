#!/bin/bash

echo "Setting memory allocation strategy..."
export TF_GPU_ALLOCATOR=cuda_malloc_async

echo "Enabling XLA for optimized computations..."
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices"

echo "Executing the training script using pipenv..."
pipenv run python training.py
