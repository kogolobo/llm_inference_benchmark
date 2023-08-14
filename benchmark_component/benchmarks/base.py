from dataclasses import dataclass
from abc import ABC, abstractmethod
from datasets import Dataset
from tqdm import tqdm
from typing import List, Tuple, Any

@dataclass
class GenerationResult:
    prompts: List[str]
    predictions: List[str]
    time: float = 0.0
    out_tokens: int = 0
    tokens_per_sec: float = 0.0

@dataclass
class BenchmarkStats:
    avg_time: float = 0.0
    avg_out_tokens: float = 0.0
    avg_tokens_per_sec: float = 0.0

class Benchmark(ABC):
    def __init__(self, model_path: str, dataset: Dataset, args: Any) -> None:
        self.model_path = model_path
        self.dataset = dataset
        self.args = args
        self.model = None
        self.tokenizer = None
        self.results = []
        self.stats = BenchmarkStats()

    def run(self) -> Tuple[List[GenerationResult], BenchmarkStats]:

        for batch_start_idx in tqdm(range(0, len(self.dataset), self.args.batch_size)):
            batch_end_idx = min(batch_start_idx + self.args.batch_size, len(self.dataset))
            batch = self.dataset.select(range(batch_start_idx, batch_end_idx))
            try:
                results = self.generate(batch['text'])
                self.results.append(results)
            except Exception as e:
                print(f"Error generating results for batch {batch_start_idx} to {batch_end_idx}: {e}")
                continue

        self._calculate_stats()
        return self.results, self.stats
    
    @abstractmethod
    def generate(self, prompts: str) -> GenerationResult:
        raise NotImplementedError()
    
    def _calculate_stats(self) -> None:
        if len(self.results) == 0:
            return

        total_predictions = 0
        total_time = 0.0
        total_out_tokens = 0
        is_first_result = True
        for result in self.results:
            if not is_first_result:
                total_predictions += len(result.predictions)
                total_time += result.time
                total_out_tokens += result.out_tokens
            else:
                # The first result is discarded because it is usually slower
                is_first_result = False
        
        self.stats.avg_time = total_time / total_predictions
        self.stats.avg_out_tokens = total_out_tokens / total_predictions
        self.stats.avg_tokens_per_sec = total_out_tokens / total_time