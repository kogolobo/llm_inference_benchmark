import os
import logging
from dataclasses import dataclass, field
from typing import Dict

try:
    from azure.ai.ml import MLClient, load_component
    from azure.ai.ml.dsl import pipeline
    
except:
    raise("Please pip install azure-ai-ml==1.0.0")

from azure.identity import DefaultAzureCredential

import logging
import hydra
from hydra.core.config_store import ConfigStore

@dataclass
class AMLConfig:
    workspace_name: str = str()
    resource_group: str = str()
    subscription_id: str = str()
    cpu_target: str = str()
    gpu_target: str = str()
    use_local_components: bool = False

@dataclass
class RunConfig:
    experiment_name: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    priority: int = 1000

@dataclass
class PipelineConfig:
    run: RunConfig
    aml_config: AMLConfig

cs = ConfigStore.instance()
cs.store(name="config", node=PipelineConfig)

@hydra.main(config_path="configs", config_name="benchmark", version_base="1.2")
def main(config: PipelineConfig):

    logging.getLogger('azure').setLevel(logging.WARN) 

    ml_client_workspace = MLClient(
        credential=DefaultAzureCredential(),         
        subscription_id=config.aml_config.subscription_id,
        resource_group_name=config.aml_config.resource_group,
        workspace_name=config.aml_config.workspace_name,
    )

    ml_client_registry = MLClient(
        credential=DefaultAzureCredential(),  
        subscription_id=config.registry_config.subscription_id,
        resource_group_name=config.registry_config.resource_group,
        registry_name=config.registry_config.registry_name,
    )

    # load components
    
    if config.aml_config.use_local_components:
        benchmark_func = load_component(source=os.path.join(".", "benchmark_component", "component_spec.yaml"))
        
    @pipeline(
        name=config.run.job_name,
        compute=config.aml_config.cpu_target,
    )
    def sample_pipeline():
        benchmarks = []
        for benchmark_type in ['hf_32bit', 'hf_8bit', 'hf_4bit', 'vllm_gptq', 'vllm', 'exllama', 'ctranslate', 'llama_cpp']:
            benchmark = benchmark_func(
                benchmark=benchmark_type, 
                **config.benchmark
            )
            benchmark.environment_variables = {
                "HUGGINGFACE_TOKEN": os.environ.get("HUGGINGFACE_TOKEN", ""),
            }
            benchmark.compute = config.aml_config.single_gpu_target
            benchmarks.append(benchmark)


    tags = {
        "task": "benchmark_llama"
    }
    print(f"Submitting a run with tags: {tags}")
    pipeline_run = ml_client_workspace.jobs.create_or_update(
        sample_pipeline(),
        tags=tags,
        experiment_name=config.run.experiment_name
    )

    print(f"\n\nWorkspace Job link: \n{pipeline_run.studio_url}")

if __name__ == "__main__":
    logger = logging.getLogger("azure")
    logger.setLevel(logging.WARNING)
    main()
