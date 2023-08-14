# LLM Inference Benchmarking

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


