#!/bin/bash

# Adjust memory allocation strategy
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Enable XLA to use available XLA devices for optimized computations 
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices"

# Execute Jupyter Notebook using pipenv
pipenv run jupyter notebook
