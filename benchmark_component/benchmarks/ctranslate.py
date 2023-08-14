import time
import torch
from typing import Any, List
from datasets import Dataset
from transformers import LlamaTokenizerFast

# Installed with: pip install ctranslate2
from ctranslate2 import Generator

from .base import Benchmark, GenerationResult


class CtranslateBenchmark(Benchmark):
    def __init__(self, model_path: str, dataset: Dataset, args: Any) -> None:
        super().__init__(model_path, dataset, args)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = Generator(model_path, device=device, compute_type="int8_float16")
        self.tokenizer = LlamaTokenizerFast.from_pretrained(model_path)

    def generate(self, prompts: List[str]) -> GenerationResult:
        tokens = [self.tokenizer.convert_ids_to_tokens(input_ids) for input_ids in self.tokenizer(prompts)['input_ids']]
        start = time.perf_counter()
        results = self.generator.generate_batch(
            start_tokens=tokens, 
            include_prompt_in_result=False, 
            max_batch_size=self.args.batch_size,
            max_length=self.args.max_new_tokens,
            sampling_topp=self.args.top_p,
            sampling_temperature=self.args.temperature
        )
        end = time.perf_counter()

        predictions = [self.tokenizer.decode(result.sequences_ids[0]) for result in results]
        out_tokens = sum([len(result.sequences_ids[0]) for result in results])
        time_taken = end - start
        tokens_per_sec = out_tokens / time_taken

        return GenerationResult(prompts, predictions, time_taken, out_tokens, tokens_per_sec)