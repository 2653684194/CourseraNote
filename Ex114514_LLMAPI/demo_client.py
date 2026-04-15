import time
import random
from typing import List, Dict, Optional, Generator

class DemoLLMClient:
    """
    演示模式客户端 - 无需真实API即可体验完整功能
    
    使用方法：
    在 .env 中设置：OPENAI_API_KEY=demo
    然后运行 python main.py 即可
    """

    def __init__(self):
        self.total_tokens_used = 0
        self.total_requests = 0
        
        self.demo_responses = {
            "你好": "你好！我是Demo模式的AI助手。这是一个模拟响应，无需真实API即可体验完整功能。\n\n💡 提示：要使用真实的AI，请查看 FREE_API_SOLUTIONS.py 获取免费方案！",
            
            "默认": f"""
🎭 **这是演示模式响应**

你发送的消息已被接收并处理（模拟）。

**当前状态：**
- ✅ 配置加载成功
- ✅ 对话管理正常工作  
- ✅ 界面渲染完美
- ⚠️  使用的是模拟AI响应

**可用功能测试：**
1. 尝试输入 `/stats` 查看统计信息
2. 辝入 `/clear` 清空对话历史
3. 输入 `/export` 导出对话记录
4. 输入 `/system` 修改系统提示词

**如何切换到真实AI？**
━━━━━━━━━━━━━━━━━━━━━━━━━
🦙 **Ollama本地模型（推荐）**
   - 完全免费、无限使用
   - 运行：python FREE_API_SOLUTIONS.py
   
⚡ **Groq免费云API**
   - 超快速度、有免费额度
   - 访问：https://console.groq.com/keys
   
☁️ **OpenAI官方API**
   - 需付费但质量最好
   - 访问：https://platform.openai.com
━━━━━━━━━━━━━━━━━━━━━━━━━

💡 *这个演示模式让你无需任何配置即可体验所有功能界面！*
""",
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict:
        """模拟API调用"""
        
        self.total_requests += 1
        
        user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_message = msg["content"]
                break
        
        time.sleep(0.5) 
        
        response_content = self._generate_response(user_message)
        
        if stream:
            return self._stream_response(response_content)
        
        usage = {
            "prompt_tokens": len(user_message),
            "completion_tokens": len(response_content),
            "total_tokens": len(user_message) + len(response_content)
        }
        
        self.total_tokens_used += usage["total_tokens"]
        
        result = {
            "content": response_content,
            "model": "demo-mode-v1",
            "usage": usage
        }
        
        return result

    def _generate_response(self, user_message: str) -> str:
        """根据用户消息生成模拟回复"""
        
        message_lower = user_message.lower().strip()
        
        for key, response in self.demo_responses.items():
            if key in message_lower or key == "默认":
                if key != "默认" and key in message_lower:
                    return response
        
        variations = [
            f"""🤔 **关于「{user_message[:30]}...」的思考**

这是一个很好的问题！在演示模式下，我无法提供真实的AI回答，但我可以展示系统的所有功能。

**系统功能验证清单：**
✅ 消息接收和处理  
✅ 上下文管理（多轮对话）  
✅ Token计数和统计  
✅ 流式输出支持  
✅ 错误处理机制  

💡 **下一步建议：**
运行 `python FREE_API_SOLUTIONS.py` 查看如何获取免费的AI API密钥，或安装 Ollama 使用完全免费的本地模型！

当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}""",

            f"""✨ **收到你的消息！**

消息内容：「{user_message[:50]}{'...' if len(user_message) > 50 else ''}」

在演示模式下，这只是一个模拟的响应示例。实际使用时，这里会显示AI的真实回答。

**你可以尝试：**
• 问一些问题测试对话流程
• 使用 /clear 命令清空历史
• 使用 /stats 查看统计数据
• 使用 /export 导出对话

🚀 准备好使用真实AI了吗？
→ 查看 FREE_API_SOLUTIONS.py 获取免费方案！"""
        ]
        
        return random.choice(variations)

    def _stream_response(self, content: str) -> Generator[str, None, None]:
        """模拟流式输出"""
        words = content.split(' ')
        for word in words:
            yield word + ' '
            time.sleep(0.02) 

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
            "model": "demo-mode-v1 (simulation)",
            "mode": "DEMO"
        }
