from typing import List, Dict, Optional
from datetime import datetime

class ConversationManager:
    def __init__(self, system_prompt: Optional[str] = None, max_history: int = 10):
        self.messages: List[Dict[str, str]] = []
        self.system_prompt = system_prompt or "你是一个有帮助的AI助手。"
        self.max_history = max_history
        self.created_at = datetime.now()
        self.message_count = 0
        
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self.message_count += 1
        self._trim_history()

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self.message_count += 1

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages.copy()

    def _trim_history(self) -> None:
        if len(self.messages) > self.max_history * 2 + 1:
            system_msg = self.messages[0] if self.messages and self.messages[0]["role"] == "system" else None
            user_assistant_msgs = [m for m in self.messages if m["role"] != "system"]
            keep_recent = user_assistant_msgs[-(self.max_history * 2):]
            
            self.messages = []
            if system_msg:
                self.messages.append(system_msg)
            self.messages.extend(keep_recent)

    def clear(self) -> None:
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
        self.message_count = 0

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = prompt
        else:
            self.messages.insert(0, {"role": "system", "content": prompt})

    def export_conversation(self) -> str:
        lines = [f"对话开始时间: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}"]
        lines.append("=" * 50)
        
        for msg in self.messages:
            role = "用户" if msg["role"] == "user" else "助手" if msg["role"] == "assistant" else "系统"
            lines.append(f"\n[{role}]:")
            lines.append(msg["content"])
        
        lines.append("\n" + "=" * 50)
        return "\n".join(lines)

    def get_stats(self) -> Dict:
        return {
            "total_messages": self.message_count,
            "current_context_length": len(self.messages),
            "created_at": self.created_at.isoformat(),
            "has_system_prompt": bool(self.system_prompt)
        }
