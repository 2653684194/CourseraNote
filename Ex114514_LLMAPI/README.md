# LLM API 调用工具

一个简单但功能完善的LLM（大语言模型）API调用工具，支持OpenAI及兼容接口。

## ✨ 特性

- 🎯 **多模型支持** - 支持OpenAI GPT系列及其他兼容API
- 💬 **对话管理** - 智能的上下文管理和历史记录
- 🔄 **自动重试** - 内置指数退避重试机制
- 📊 **使用统计** - 实时Token消耗和请求统计
- 🎨 **美观界面** - 基于Rich库的终端美化输出
- 💾 **导出功能** - 支持导出完整对话记录
- ⚡ **流式响应** - 支持实时流式输出

## 📁 项目结构

```
Ex114514_LLMAPI/
├── .env                 # API密钥配置（需要修改）
├── .env.example         # 配置示例
├── requirements.txt     # Python依赖
├── config.py           # 配置管理模块
├── llm_client.py       # 核心API调用模块
├── conversation.py     # 对话管理模块
├── main.py             # 主程序入口（交互式界面）
└── examples.py         # 使用示例
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

编辑 `.env` 文件，设置你的API密钥：

```env
OPENAI_API_KEY=你的真实API密钥
```

**获取API密钥:**
- OpenAI: https://platform.openai.com/api-keys
- 或其他兼容服务提供商

### 3. 运行程序

#### 交互式模式（推荐）
```bash
python main.py
```

#### 示例模式
```bash
python examples.py 1   # 简单纯文本对话
python examples.py 2   # 多轮对话
python examples.py 3   # 流式响应
python examples.py 4   # 自定义配置
```

## 🎮 使用指南

### 交互式命令

在交互式界面中，可以使用以下命令：

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助信息 |
| `/clear` | 清空对话历史 |
| `/export` | 导出当前对话到文件 |
| `/stats` | 显示使用统计信息 |
| /system `<提示>` | 设置系统提示词 |
| `/quit` 或 `/exit` | 退出程序 |

### 编程方式使用

```python
from config import LLMConfig
from llm_client import LLMClient
from conversation import ConversationManager

# 加载配置
config = LLMConfig.from_env()

# 创建客户端
client = LLMClient(config)

# 简单对话
response = client.simple_chat("你好！")
print(response)

# 或使用对话管理器
conv = ConversationManager(system_prompt="你是一个有帮助的助手")
conv.add_user_message("解释一下机器学习")
response = client.chat_completion(conv.get_messages())
print(response['content'])
```

## ⚙️ 配置说明

### .env 配置项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | API密钥（必填） | - |
| `OPENAI_BASE_URL` | API基础URL | https://api.openai.com/v1 |
| `DEFAULT_MODEL` | 默认模型 | gpt-4o-mini |
| `MAX_TOKENS` | 最大Token数 | 2048 |
| `TEMPERATURE` | 温度参数（0-2） | 0.7 |
| `MAX_RETRIES` | 最大重试次数 | 3 |
| `RETRY_DELAY` | 重试延迟（秒） | 1 |

### 支持的其他API服务

只需修改 `.env` 中的 `OPENAI_BASE_URL` 即可切换到兼容服务：

- **Azure OpenAI**: `https://your-resource.openai.azure.com/openai/deployments/your-deployment`
- **本地模型 (Ollama)**: `http://localhost:11434/v1`
- **其他兼容服务**: 参考对应文档

## 🔧 高级功能

### 自定义配置

```python
from config import LLMConfig

custom_config = LLMConfig(
    api_key="your-key",
    base_url="https://api.example.com/v1",
    model="custom-model",
    max_tokens=4096,
    temperature=0.9,
    max_retries=5,
    retry_delay=2.0
)
```

### 流式响应

```python
messages = [{"role": "user", "content": "写一个故事"}]

for chunk in client.chat_completion(messages, stream=True):
    print(chunk, end='', flush=True)
```

### 对话导出

```python
conversation_text = conv.export_conversation()
with open("chat.txt", "w", encoding="utf-8") as f:
    f.write(conversation_text)
```

## 📊 使用统计

程序会自动跟踪：
- 总请求数
- Token消耗量
- 对话消息数
- 上下文长度

使用 `/stats` 命令或调用 `client.get_stats()` 查看。

## 🔒 安全提示

1. **不要提交API密钥** - `.env` 文件已在 `.gitignore` 中排除
2. **使用环境变量** - 生产环境建议通过环境变量传递密钥
3. **定期轮换密钥** - 定期更新API密钥以保证安全

## ❓ 常见问题

### Q: 出现"API Key未配置"错误
A: 请确保已正确编辑 `.env` 文件并填入有效的API密钥

### Q: 如何使用其他模型？
A: 修改 `.env` 中的 `DEFAULT_MODEL` 为支持的模型名称

### Q: 速率限制错误？
A: 程序会自动重试，也可增加 `MAX_RETRIES` 和 `RETRY_DELAY`

### Q: Token超限怎么办？
A: 调整 `MAX_TOKENS` 或使用 `/clear` 清空对话历史

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

**享受与AI对话的乐趣吧！🚀**
