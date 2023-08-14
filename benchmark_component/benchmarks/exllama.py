import os, glob
import torch

import time
from typing import Any, List
from datasets import Dataset

# Installed with: pip install git+https://github.com/jllllll/exllama
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator

from .base import Benchmark, GenerationResult

class ExllamaBenchmark(Benchmark):
    MIN_TEMPERATURE = 1e-5

    def __init__(self, model_path: str, dataset: Dataset, args: Any) -> None:
        super().__init__(model_path, dataset, args)
        self._load_model()

    def _load_model(self):
        tokenizer_path = os.path.join(self.model_path, "tokenizer.model")
        model_config_path = os.path.join(self.model_path, "config.json")
        st_pattern = os.path.join(self.model_path, "*.safetensors")
        checkpoint_path = glob.glob(st_pattern)[0]

        config = ExLlamaConfig(model_config_path)               
        config.model_path = checkpoint_path                          

        model = ExLlama(config)                                 
        tokenizer = ExLlamaTokenizer(tokenizer_path)            

        cache = ExLlamaCache(model, batch_size=self.args.batch_size)
        self.generator = ExLlamaGenerator(model, tokenizer, cache) 
        self.generator.disallow_tokens([tokenizer.eos_token_id])

        self.generator.settings.token_repetition_penalty_max = 1.2
        self.generator.settings.typical = 0.0 # disable locally typical sampling
        self.generator.settings.top_p = self.args.top_p

        # Greedy decoding when temperature approaches 0
        if self.args.temperature <= self.MIN_TEMPERATURE:
            self.generator.settings.temperature = 1.0
            self.generator.settings.top_k = 1

    def generate(self, prompts: List[str]) -> GenerationResult:
        start = time.perf_counter()
        results = self.generator.generate_simple(prompts, max_new_tokens=self.args.max_new_tokens)
        end = time.perf_counter()


        if isinstance(results, str):
            results = [results]
            
        predictions = [result.replace(prompt, "") for prompt, result in zip(prompts, results)]
        out_tokens = torch.numel(self.generator.tokenizer.encode(predictions)[:, 1:])
        time_taken = end - start
        tokens_per_sec = out_tokens / time_taken
        

        return GenerationResult(prompts, predictions, time_taken, out_tokens, tokens_per_sec)