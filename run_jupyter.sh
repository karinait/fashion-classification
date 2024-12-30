#!/bin/bash

# Adjust memory allocation strategy
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Execute Jupyter Notebook using pipenv
pipenv run jupyter notebook
