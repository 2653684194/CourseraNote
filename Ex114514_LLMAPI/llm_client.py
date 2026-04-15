import time
from typing import List, Dict, Optional, Generator
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from config import LLMConfig

class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            max_retries=config.max_retries
        )
        self.total_tokens_used = 0
        self.total_requests = 0

    def _retry_with_backoff(self, func, *args, **kwargs):
        last_exception = None
        for attempt in range(self.config.max_retries):
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                last_exception = e
                wait_time = self.config.retry_delay * (2 ** attempt)
                print(f"⚠️ 速率限制，等待 {wait_time:.1f}秒 后重试... (尝试 {attempt + 1}/{self.config.max_retries})")
                time.sleep(wait_time)
            except APIConnectionError as e:
                last_exception = e
                wait_time = self.config.retry_delay * (attempt + 1)
                print(f"⚠️ 连接错误，等待 {wait_time:.1f}秒 后重试... (尝试 {attempt + 1}/{self.config.max_retries})")
                time.sleep(wait_time)
            except APIError as e:
                if e.status_code and e.status_code >= 500:
                    last_exception = e
                    wait_time = self.config.retry_delay * (attempt + 1)
                    print(f"⚠️ 服务器错误 ({e.status_code})，等待 {wait_time:.1f}秒 后重试...")
                    time.sleep(wait_time)
                else:
                    raise
        
        raise Exception(f"❌ 重试{self.config.max_retries}次后仍然失败: {last_exception}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict:
        self.total_requests += 1
        
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": stream
        }

        if stream:
            return self._stream_response(params)
        
        response = self._retry_with_backoff(
            self.client.chat.completions.create,
            **params
        )
        
        result = {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
        self.total_tokens_used += result["usage"]["total_tokens"]
        return result

    def _stream_response(self, params: Dict) -> Generator[str, None, None]:
        stream = self._retry_with_backoff(
            self.client.chat.completions.create,
            **params
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def simple_chat(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        
        result = self.chat_completion(messages)
        return result["content"]

    def get_stats(self) -> Dict:
        return {
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "model": self.config.model
        }
