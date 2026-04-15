"""
LLM API 调用工具 - 使用示例
演示如何在不同场景下使用该工具
"""

from config import LLMConfig
from llm_client import LLMClient
from conversation import ConversationManager

def example_simple_chat():
    """示例1: 简单的单轮对话"""
    print("=" * 50)
    print("示例1: 简单单轮对话")
    print("=" * 50)
    
    # 从环境变量加载配置
    config = LLMConfig.from_env()
    
    # 创建客户端
    client = LLMClient(config)
    
    # 发送简单消息
    response = client.simple_chat(
        user_message="用一句话解释什么是人工智能",
        system_prompt="你是一个专业的技术顾问"
    )
    
    print(f"AI回复: {response}")
    print(f"\n使用统计: {client.get_stats()}")

def example_multi_turn_conversation():
    """示例2: 多轮对话（带上下文管理）"""
    print("\n" + "=" * 50)
    print("示例2: 多轮对话")
    print("=" * 50)
    
    config = LLMConfig.from_env()
    client = LLMClient(config)
    
    # 创建对话管理器，设置系统提示词和最大历史条数
    conv = ConversationManager(
        system_prompt="你是一个有帮助的Python编程助手",
        max_history=10
    )
    
    # 第一轮对话
    conv.add_user_message("什么是列表推导式？")
    response1 = client.chat_completion(conv.get_messages())
    conv.add_assistant_message(response1['content'])
    print(f"Q1: 什么是列表推导式？")
    print(f"A1: {response1['content'][:100]}...\n")
    
    # 第二轮对话（会包含上下文）
    conv.add_user_message("能给我一个具体的例子吗？")
    response2 = client.chat_completion(conv.get_messages())
    conv.add_assistant_message(response2['content'])
    print(f"Q2: 能给我一个具体的例子吗？")
    print(f"A2: {response2['content'][:100]}...\n")
    
    # 查看对话统计
    print(f"对话统计: {conv.get_stats()}")

def example_streaming():
    """示例3: 流式响应"""
    print("\n" + "=" * 50)
    print("示例3: 流式响应")
    print("=" * 50)
    
    config = LLMConfig.from_env()
    client = LLMClient(config)
    
    messages = [
        {"role": "user", "content": "写一首关于编程的短诗"}
    ]
    
    print("AI正在生成（流式输出）:")
    print("-" * 30)
    
    # 使用流式响应
    for chunk in client.chat_completion(messages, stream=True):
        print(chunk, end='', flush=True)
    
    print("\n" + "-" * 30)

def example_custom_config():
    """示例4: 自定义配置"""
    print("\n" + "=" * 50)
    print("示例4: 自定义配置")
    print("=" * 50)
    
    # 手动创建配置（不从环境变量）
    custom_config = LLMConfig(
        api_key="your-api-key-here",  # 替换为你的API密钥
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        max_tokens=1024,
        temperature=0.5,
        max_retries=5,
        retry_delay=2.0
    )
    
    print(f"自定义配置已创建:")
    print(f"  - 模型: {custom_config.model}")
    print(f"  - 最大Token: {custom_config.max_tokens}")
    print(f"  - 温度: {custom_config.temperature}")
    print(f"  - 最大重试次数: {custom_config.max_retries}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example_simple_chat()
        elif example_num == "2":
            example_multi_turn_conversation()
        elif example_num == "3":
            example_streaming()
        elif example_num == "4":
            example_custom_config()
        else:
            print("用法: python examples.py [1-4]")
            print("  1 - 简单纯文本对话")
            print("  2 - 多轮对话（带上下文）")
            print("  3 - 流式响应")
            print("  4 - 自定义配置示例")
    else:
        print("请选择要运行的示例:")
        print("  python examples.py 1  # 运行示例1")
        print("  python examples.py 2  # 运行示例2")
        print("  python examples.py 3  # 运行示例3")
        print("  python examples.py 4  # 运行示例4")
