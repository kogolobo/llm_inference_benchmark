import argparse
import os
import json
import transformers
import subprocess
import shutil
import glob

from dataclasses import asdict
from functools import partial
from azureml.core import Run
from datasets import load_dataset
from huggingface_hub import snapshot_download, hf_hub_download

from args import GenerationArguments, DataArguments
from benchmarks import (
    HuggingFaceBenchmark, 
    VllmBenchmark, 
    ExllamaBenchmark, 
    CtranslateBenchmark, 
    LLamaCppBenchmark
)

def run_benchmark(benchmark_cls, dataset, args):
    run = Run.get_context()
    benchmark = benchmark_cls(dataset=dataset, args=args)
    results, stats = benchmark.run()
    for key, value in asdict(stats).items():
        run.log(key, value)
    with open(os.path.join(args.output_path, "stats.json"), "w") as f:
        json.dump(asdict(stats), f, indent=2)
    with open(os.path.join(args.output_path, "results.json"), "w") as f:
        results = [asdict(result) for result in results]
        json.dump(results, f, indent=2)
    return stats

def prepare_models(output_path: str, benchmark: str):
    prepare_hf = False
    prepare_gptq = False
    prepare_ggml = False
    prepare_ctranslate = False
    
    if 'hf' or 'vllm' in benchmark:
        prepare_hf = True
    if  'gptq' or 'exllama' in benchmark:
        prepare_gptq = True
    if 'ctranslate' in benchmark:
        prepare_hf = True
        prepare_ctranslate = True
    if 'llama_cpp' in benchmark:
        prepare_ggml = True

    hf_model_path = None
    if prepare_hf:
        hf_model_path = os.path.join(output_path, "meta_llama_Llama_2_7b_hf")
        os.makedirs(hf_model_path, exist_ok=True)
        snapshot_download('meta-llama/Llama-2-7b-hf', local_dir=hf_model_path, local_dir_use_symlinks=False)

    gptq_model_path = None
    if prepare_gptq:
        gptq_model_path = os.path.join(output_path, "llama_2_7b_gptq")
        os.makedirs(gptq_model_path, exist_ok=True)
        snapshot_download('TheBloke/Llama-2-7B-GPTQ', local_dir=gptq_model_path, local_dir_use_symlinks=False)

    ctranslate_model_path = None
    if prepare_ctranslate:
        ctranslate_model_path = os.path.join(output_path, "llama_2_7b_ctranslate")
        os.makedirs(ctranslate_model_path, exist_ok=True)
        res = subprocess.run(f"ct2-transformers-converter --model {hf_model_path} --quantization int8_float16 --output_dir {ctranslate_model_path} --force", shell=True, capture_output=True)
        if res.returncode != 0:
            print(res.stdout.decode())
            print(res.stderr.decode())
            ctranslate_model_path = None
        else:
            # Copy tokenizer files to ctranslate model path
            for file in glob.glob(os.path.join(hf_model_path, "*tokenizer*")):
                shutil.copy(file, ctranslate_model_path)

    ggml_path = None
    if prepare_ggml:
        ggml_dir = os.path.join(output_path, "llama_2_7b_ggml")
        os.makedirs(ggml_dir, exist_ok=True)
        hf_hub_download("TheBloke/Llama-2-7B-GGML", filename="llama-2-7b.ggmlv3.q2_K.bin", local_dir=ggml_dir, local_dir_use_symlinks=False)
        ggml_path = os.path.join(ggml_dir, "llama-2-7b.ggmlv3.q2_K.bin")

    return hf_model_path, gptq_model_path, ctranslate_model_path, ggml_path


def main() -> None:
    hfparser = transformers.HfArgumentParser((GenerationArguments, DataArguments))
    generation_args, data_args, extra_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(**vars(generation_args), **vars(data_args))
    
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    if args.max_eval_samples is not None:
        dataset = dataset.select(range(args.max_eval_samples))

    hf_model_path, gptq_model_path, ctranslate_model_path, ggml_path = prepare_models(args.output_path, args.benchmark)
    benchmarks = {
        "hf_32bit": partial(HuggingFaceBenchmark, model_path=hf_model_path, bits=32),
        "hf_8bit": partial(HuggingFaceBenchmark, model_path=hf_model_path, bits=8),
        "hf_4bit": partial(HuggingFaceBenchmark, model_path=hf_model_path, bits=4),
        "vllm_gptq": partial(VllmBenchmark, model_path=gptq_model_path),
        "vllm": partial(VllmBenchmark, model_path=hf_model_path),
        "exllama": partial(ExllamaBenchmark, model_path=gptq_model_path),
        "ctranslate": partial(CtranslateBenchmark, model_path=ctranslate_model_path) if ctranslate_model_path is not None else None,
        "llama_cpp": partial(LLamaCppBenchmark, model_path=ggml_path),
    }
    run_benchmark(benchmarks[args.benchmark], dataset, args)

if __name__ == "__main__":
    main()