from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from config import LLMConfig
from llm_client import LLMClient
from demo_client import DemoLLMClient
from conversation import ConversationManager
import sys

console = Console()

def print_banner():
    banner = """
[bold cyan]
╔═══════════════════════════════════════╗
║     🤖 LLM API 调用工具 v1.1         ║
║   支持 OpenAI/Ollama/演示模式        ║
╚═══════════════════════════════════════╝
[/bold cyan]
"""
    console.print(Panel(banner, border_style="cyan"))

def print_help():
    help_text = """
[bold]可用命令:[/bold]
  [green]/help[/green]       - 显示此帮助信息
  [green]/clear[/green]      - 清空对话历史
  [green]/export[/green]     - 导出当前对话
  [green]/stats[/green]      - 显示使用统计
  [green]/system <提示>[/green] - 设置系统提示词
  [green]/mode[/green]       - 显示当前运行模式
  [green]/quit[/green] 或 [green]/exit[/green] - 退出程序
  
[bold]使用说明:[/bold]
  • 直接输入文字即可与AI对话
  • 支持多行输入（Shift+Enter换行，Enter发送）
  • 输入内容会自动保存到对话历史中
    """
    console.print(Panel(help_text, title="📖 帮助", border_style="green"))

def print_stats(client, conv: ConversationManager):
    client_stats = client.get_stats()
    conv_stats = conv.get_stats()
    
    mode_info = ""
    if hasattr(client_stats, 'mode') or (isinstance(client_stats, dict) and 'mode' in client_stats):
        mode_info = f"\n  • 运行模式: {client_stats.get('mode', 'NORMAL')}"
    
    stats_text = f"""
[bold]API 使用统计:[/bold]
  • 总请求数: {client_stats['total_requests']}
  • 总Token消耗: {client_stats['total_tokens_used']}
  • 当前模型: {client_stats['model']}{mode_info}

[bold]对话统计:[/bold]
  • 消息总数: {conv_stats['total_messages']}
  • 上下文长度: {conv_stats['current_context_length']}
  • 创建时间: {conv_stats['created_at'][:19]}
    """
    console.print(Panel(stats_text, title="📊 统计信息", border_style="yellow"))

def create_client(config: LLMConfig):
    """根据配置创建合适的客户端"""
    if config.is_demo_mode:
        return DemoLLMClient(), "🎭 演示模式"
    else:
        return LLMClient(config), config.get_mode_description()

def main():
    try:
        print_banner()
        console.print("[dim]正在加载配置...[/dim]")
        
        config = LLMConfig.from_env()
        client, mode_desc = create_client(config)
        conv = ConversationManager()
        
        console.print(f"[green]✓ 配置加载成功！[/green]")
        console.print(f"[bold cyan]{mode_desc}[/bold cyan]")
        
        if config.is_ollama:
            console.print("[yellow]💡 提示：确保 Ollama 已启动并下载了模型[/yellow]")
            console.print("[dim]   安装指南：python FREE_API_SOLUTIONS.py[/dim]")
        elif config.is_demo_mode:
            console.print("[magenta]🎭 当前为演示模式，使用模拟响应[/magenta]")
            console.print("[dim]   切换到真实AI：编辑 .env 文件或查看 FREE_API_SOLUTIONS.py[/dim]")
        
        console.print(f"[dim]最大Token: {config.max_tokens} | 温度: {config.temperature}[/dim]")
        console.print()
        print_help()
        console.print("\n[bold cyan]开始对话吧！（输入 /help 查看命令）[/bold cyan]\n")
        
        while True:
            try:
                user_input = console.input("[bold blue]你: [/bold blue]")
                
                if not user_input.strip():
                    continue
                
                if user_input.startswith('/'):
                    cmd = user_input.lower().strip()
                    
                    if cmd in ['/quit', '/exit', '/q']:
                        console.print("[yellow]再见！👋[/yellow]")
                        break
                    elif cmd == '/help':
                        print_help()
                    elif cmd == '/clear':
                        conv.clear()
                        console.print("[green]✓ 对话历史已清空[/green]")
                    elif cmd == '/export':
                        exported = conv.export_conversation()
                        timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"conversation_{timestamp}.txt"
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(exported)
                        console.print(f"[green]✓ 对话已导出到: {filename}[/green]")
                    elif cmd == '/stats':
                        print_stats(client, conv)
                    elif cmd == '/mode':
                        console.print(Panel(mode_desc, title="ℹ️ 当前模式", border_style="blue"))
                    elif cmd.startswith('/system '):
                        system_prompt = cmd[8:].strip()
                        if system_prompt:
                            conv.set_system_prompt(system_prompt)
                            console.print(f"[green]✓ 系统提示词已更新[/green]")
                        else:
                            console.print("[red]❌ 请提供系统提示词内容[/red]")
                    else:
                        console.print(f"[red]❌ 未知命令: {cmd}[/red]")
                        console.print("[dim]输入 /help 查看可用命令[/dim]")
                    continue
                
                conv.add_user_message(user_input)
                
                status_msg = "[bold green]AI正在思考...[/bold green]"
                if config.is_demo_mode:
                    status_msg = "[bold magenta]Demo模式生成响应中...[/bold magenta]"
                
                with console.status(status_msg):
                    response = client.chat_completion(conv.get_messages())
                
                conv.add_assistant_message(response['content'])
                
                console.print("\n[bold magenta]AI: [/bold magenta]")
                
                try:
                    md = Markdown(response['content'])
                    console.print(md)
                except:
                    console.print(response['content'])
                
                usage = response['usage']
                console.print(f"\n[dim]└─ Token使用: {usage['prompt_tokens']} (输入) + {usage['completion_tokens']} (输出) = {usage['total_tokens']} (总计)[/dim]\n")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]\n检测到中断信号...[/yellow]")
                choice = console.input("确定要退出吗？(y/n): ")
                if choice.lower() in ['y', 'yes']:
                    break
                console.print("[green]继续对话...[/green]\n")
            except Exception as e:
                error_msg = str(e)
                if "ollama" in error_msg.lower() and "connection" in error_msg.lower():
                    console.print("\n[red]❌ 无法连接到 Ollama 服务！[/red]")
                    console.print("[yellow]请确保：[/yellow]")
                    console.print("  1. Ollama 已安装并运行")
                    console.print("  2. 已下载模型：ollama pull qwen2.5:3b")
                    console.print("  3. 服务地址正确：http://localhost:11434")
                    console.print("\n[dim]安装指南：python FREE_API_SOLUTIONS.py[/dim]")
                    console.print("[dim]或切换到演示模式：设置 OPENAI_API_KEY=demo[/dim]\n")
                else:
                    console.print(f"\n[red]❌ 错误: {error_msg}[/red]\n")
    
    except ValueError as e:
        console.print(f"[red]{str(e)}[/red]")
        
        console.print("\n[bold cyan]快速开始选项：[/bold cyan]\n")
        console.print("1️⃣  [green]立即体验（无需任何配置）[/green]")
        console.print("    运行：[dim]python FREE_API_SOLUTIONS.py[/dim] 查看免费方案\n")
        
        console.print("2️⃣  [green]使用演示模式[/green]")
        console.print("    编辑 .env 设置：[dim]OPENAI_API_KEY=demo[/dim]\n")
        
        console.print("3️⃣  [green]安装Ollama本地模型[/green]")
        console.print("    访问：[dim]https://ollama.com/download[/dim]\n")
        
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ 程序启动失败: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
