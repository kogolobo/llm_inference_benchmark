import time
from datasets import Dataset
from typing import Any, List

# Installed with: pip install vllm
from vllm import SamplingParams, LLM

from .base import GenerationResult, Benchmark

class VllmBenchmark(Benchmark):
    def __init__(self, model_path: str, dataset: Dataset, args: Any) -> None:
        super().__init__(model_path, dataset, args)
        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )
        self.model = LLM(model_path)

    def generate(self, prompts: List[str]) -> GenerationResult:
        start = time.perf_counter()
        results = self.model.generate(prompts, self.sampling_params)
        end = time.perf_counter()


        out_tokens = sum([len(result.outputs[0].token_ids) for result in results])
        time_taken = end - start
        tokens_per_sec = out_tokens / time_taken
        predictions = [result.outputs[0].text for result in results]

        return GenerationResult(prompts, predictions, time_taken, out_tokens, tokens_per_sec)