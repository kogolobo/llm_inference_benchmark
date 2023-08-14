from dataclasses import dataclass, field
from typing import Optional
@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default="roneneldan/TinyStories",
        metadata={"help": "Which dataset to benchmark on."}
    )

    dataset_split: Optional[str] = field(
        default="validation",
        metadata={"help": "Which split of the dataset to use."}
    )

    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    output_path: Optional[str] = field(
        default=".",
        metadata={"help": "Path to save the benchmark results to."}
    )

    benchmark: Optional[str] = field(
        default="hf",
        metadata={"help": "Which benchmark to run. Options: 'hf_32bit', 'hf_8bit', 'hf_4bit', 'vllm_gptq', 'vllm', 'exllama', 'ctranslate', 'llama_cpp'"}
    )

@dataclass
class GenerationArguments:
    batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "Batch size (per device) for evaluation and prediction loops."}
    )

    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    temperature: Optional[float] = field(default=0.0)
    top_p: Optional[float] = field(default=1.0)