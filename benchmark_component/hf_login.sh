#!/usr/bin/env bash

if [ -n "$HUGGINGFACE_TOKEN" ]; then 
    huggingface-cli login --token $HUGGINGFACE_TOKEN || echo "Huggingface token is invalid. Please check your token and try again."
else
    echo "HUGGINGFACE_TOKEN is not set."
fi