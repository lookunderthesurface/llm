import os
from openai import OpenAI
from typing import Optional

class DeepSeekEngine:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._client = None
        return cls._instance
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if not hasattr(self, '_initialized'):
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "sk-5a2facdcfcfc48b2a1e9c87a21d1629f") ### api_key
            self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self._initialized = True
    
    def _init_client(self):
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
    
    def generate(self, prompt, temperature=0.7, top_p=0.9, max_tokens=256):
        self._init_client()
        
        completion = self._client.chat.completions.create(
            model="deepseek-r1-0528",
            messages=[{'role': 'user', 'content': prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content

if __name__ == "__main__":
    engine = DeepSeekEngine(
        api_key="sk-5a2facdcfcfc48b2a1e9c87a21d1629f",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    print(engine.generate("你好，请介绍一下你自己"))