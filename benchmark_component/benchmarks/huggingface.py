import warnings
from typing import Any, List
import torch
import time
from transformers import AutoModelForCausalLM, LlamaTokenizerFast, GenerationConfig
from transformers.utils import PaddingStrategy
from datasets import Dataset

from .base import Benchmark, GenerationResult

class HuggingFaceBenchmark(Benchmark):
    def __init__(self, model_path: str, dataset: Dataset, args: Any, bits: int) -> None:
        super().__init__(model_path, dataset, args)
        self.model = self._load_model(model_path, bits)
        self.tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generation_config = GenerationConfig(**vars(args))

    def _load_model(self, model_path: str, bits: int):
        if bits == 8:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_8bit=True)
        elif bits == 4:
            cuda_major, _ = torch.cuda.get_device_capability()
            if cuda_major < 8:
                warnings.warn("4-bit inference is only supported on GPUs with compute capability >= 8.0. Defaulting to 8-bit inference.")
                model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_8bit=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_4bit=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path)
        
        if torch.cuda.is_available() and bits not in [4, 8]:
            model = model.to(torch.device("cuda"))

        return model

    def generate(self, prompts: List[str]) -> GenerationResult:
        inputs = self.tokenizer(prompts, padding=PaddingStrategy.LONGEST, return_tensors="pt")
        inputs.pop("token_type_ids", None)
        for key, value in inputs.items():
            inputs[key] = value.to(self.model.device)

        start = time.perf_counter()
        results = self.model.generate(**inputs, generation_config=self.generation_config)
        end = time.perf_counter()

        out_tokens = torch.numel(results) - torch.numel(inputs["input_ids"])
        time_taken = end - start
        tokens_per_sec = out_tokens / time_taken

        predictions = self.tokenizer.batch_decode(results[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return GenerationResult(prompts, predictions, time_taken, out_tokens, tokens_per_sec)