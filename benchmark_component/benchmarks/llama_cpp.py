import time
import torch
from typing import Any, List
from datasets import Dataset
from transformers import LlamaTokenizerFast

# Installed with: CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
from llama_cpp import Llama

from .base import Benchmark, GenerationResult
from tqdm import tqdm

class LLamaCppBenchmark(Benchmark):
    def __init__(self, model_path: str, dataset: Dataset, args: Any) -> None:
        super().__init__(model_path, dataset, args)
        self.model = Llama(model_path=model_path, n_gpu_layers=50, n_ctx=2048)

    def generate(self, prompts: List[str]) -> GenerationResult:
        time_taken = 0
        results = []
        for prompt in tqdm(prompts):
            start = time.perf_counter()
            result = self.model(prompt, max_tokens=self.args.max_new_tokens, stop=None, echo=False)
            end = time.perf_counter()
            results.append(result)
            time_taken += end - start


        out_tokens = sum([result['usage']['completion_tokens'] for result in results])
        # time_taken = end - start
        tokens_per_sec = out_tokens / time_taken
        predictions = [result['choices'][0]['text'] for result in results]

        return GenerationResult(prompts, predictions, time_taken, out_tokens, tokens_per_sec)