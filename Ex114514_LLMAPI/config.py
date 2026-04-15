import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

load_dotenv()

@dataclass
class LLMConfig:
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    max_tokens: int = 2048
    temperature: float = 0.7
    max_retries: int = 3
    retry_delay: float = 1.0

    @property
    def is_ollama(self) -> bool:
        return self.api_key.lower() == 'ollama'

    @property
    def is_demo_mode(self) -> bool:
        return self.api_key.lower() == 'demo'

    @classmethod
    def from_env(cls) -> 'LLMConfig':
        api_key = os.getenv('OPENAI_API_KEY', '')
        
        if not api_key or api_key == 'your_api_key_here':
            raise ValueError(
                "\n❌ API Key未配置！\n\n"
                "🎉 别担心！我们有免费方案：\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "1️⃣  【推荐】Ollama本地模型 - 完全免费\n"
                "     运行：python FREE_API_SOLUTIONS.py\n"
                "     查看 Ollama 安装指南\n\n"
                "2️⃣  Groq免费API - 超快速\n"
                "     访问：https://console.groq.com/keys\n\n"
                "3️⃣  演示模式（无需API）\n"
                "     设置 OPENAI_API_KEY=demo\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "💡 详细指南请运行：python FREE_API_SOLUTIONS.py"
            )
        
        return cls(
            api_key=api_key,
            base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
            model=os.getenv('DEFAULT_MODEL', 'gpt-4o-mini'),
            max_tokens=int(os.getenv('MAX_TOKENS', '2048')),
            temperature=float(os.getenv('TEMPERATURE', '0.7')),
            max_retries=int(os.getenv('MAX_RETRIES', '3')),
            retry_delay=float(os.getenv('RETRY_DELAY', '1'))
        )

    def validate(self) -> bool:
        if self.is_ollama or self.is_demo_mode:
            return True
        if not self.api_key:
            return False
        if self.temperature < 0 or self.temperature > 2:
            return False
        if self.max_tokens <= 0:
            return False
        return True

    def get_mode_description(self) -> str:
        if self.is_ollama:
            return f"🦙 Ollama本地模式 | 模型: {self.model}"
        elif self.is_demo_mode:
            return "🎭 Demo演示模式（模拟响应）"
        else:
            return f"☁️ API模式 | 模型: {self.model} | 端点: {self.base_url}"
