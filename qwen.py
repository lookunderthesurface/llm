import os
import torch
from vllm import LLM, SamplingParams

class LLMEngine:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._engine = None
        return cls._instance
    
    def __init__(self, model_path=None, tensor_parallel_size=2, gpu_memory_utilization=0.92):
        if not hasattr(self, '_initialized'):
            self.model_path = model_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/Qwen2.5-7B-Instruct") ### model_path
            self.tensor_parallel_size = tensor_parallel_size
            self.gpu_memory_utilization = gpu_memory_utilization
            self._initialized = True
    
    def _init_engine(self):
        if self._engine is None:
            self._engine = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True
            )
    
    def generate(self, prompt, temperature=0.7, top_p=0.9, max_tokens=256):
        self._init_engine()
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        outputs = self._engine.generate([prompt], sampling_params=sampling_params)
        return outputs[0].outputs[0].text

if __name__ == "__main__":
    engine = LLMEngine(
        model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/Qwen2.5-7B-Instruct"),
        tensor_parallel_size=2
    )
    print(engine.generate("你好，请介绍一下你自己"))