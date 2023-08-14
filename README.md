# LLM Inference Benchmarking

## Overview
This repo contains the code for benchamrking of several optimzed LLM inference methods, using a mix of low-level optimizations, quantization techniques ([Bitsandbytes](https://github.com/TimDettmers/bitsandbytes), [GPTQ](https://github.com/PanQiWei/AutoGPTQ)), and [Paged Attention](https://github.com/vllm-project/vllm).

The libraries considered are as follows:

Method | Optimizations
---|---
[Huggingface Transformers](https://github.com/huggingface/transformers) | Basline
Huggingface Transformers 4/8 bits | Bitsandbytes Quantization
[Exllama](https://github.com/turboderp/exllama) | GPTQ Quantization, Low-level implementation
[CTansalte2](https://github.com/OpenNMT/CTranslate2) | Bitsandbytes Quantization, Low-level implementation
[Llama.cpp](https://github.com/abetlen/llama-cpp-python) | Custom Quantization, Low-level implementation
[vLLM](https://github.com/vllm-project/vllm) | Paged Attention
vLLM + GPTQ | GPTQ Quantization, Paged Attention

## Experiments
we use [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf) to benchmark the optimized inference techniques. In each experiment, we use first 6000 samples from the validation set of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset as context, requesting up to 200 tokens for each completion. We request completions in batches of 1, 4, 16, and 64, timing the latency. We report the average per-batch latency, as well as average throughput (tokens per second), discarding the times obtained for the first batch. We also report average GPU memory utilization. All experiments are run on a single AzureML Nvidia A100 80G GPU. 

### Batch Size 1
Since interactive applications require turn-by-turn input from the user, a lot of LLM inference use cases will require inference with batch size of 1. We present the corresponding results in the table below. We observe that vLLM offers the best throughput, albeit using almost all available GPU space. In this case, the additional GPTQ quantization does not help the vLLM throughput. Exllama appears as the most promising low-level optimization library with considerably lower GPU memory needs. 

Method | Latency (s) | Tokens | Throughput (token/s) | GPU Memory 
---|---|---|---|---
Transformers 32 bit | 3.62 | 144.06 | 39.79 | 32.9 
Transformers 8 bit | 18.8 | 139.76 | 7.43 | 3.74 
Transformers 4 bit | 3.35 | 72.88 | 21.74 | 3.05 
Exllama | 2.81 | 199.91 | 71.09 | 6.3 
CTranslate2 | 2.51 | 153.93 | 61.35 | 7.65 
Llama.cpp | 11.37 | 58.73 | 5.16 | 3.71 
vLLM | 1.76 | 142.76 | 81.32 | 70.38 
vLLM + GPTQ | 2.45 | 200 | 81.65 | 71.36 
