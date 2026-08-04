import asyncio
import atexit
import os
import socket
import subprocess
import sys
import time
import uuid

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from agent.tools.middleware import (
    monitor_tool, log_before_model, report_prompt_switch,
    trim_history, medical_guardrail,
)


# ── MCP 工具加载(模块级单例, ReactAgent 与 PipelineAgent 共享同一 MCP 进程) ──
_mcp_tools_cache: list | None = None
_mcp_server_proc: subprocess.Popen | None = None

# 全局后台事件循环: 所有 MCP 异步工具调用统一在此 loop 上执行,
# 避免每次 asyncio.run() 新建/销毁 loop 带来的开销与偶发 WinError 10054
import threading as _threading
_mcp_loop = asyncio.new_event_loop()
_mcp_loop_thread = _threading.Thread(target=_mcp_loop.run_forever, daemon=True)
_mcp_loop_thread.start()

# get_mcp_tools 初始化锁: 双会话并发首次加载时只允许一个线程启动 MCP 子进程
_mcp_init_lock = _threading.Lock()


def _start_mcp_server() -> tuple[subprocess.Popen, str]:
    """启动 mcp_server.py 子进程(streamable-http), 返回(进程, 服务地址)"""
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    env = dict(os.environ)
    env['MEDIAGENT_MCP_TRANSPORT'] = 'streamable-http'
    env['MEDIAGENT_MCP_PORT'] = str(port)

    proc = subprocess.Popen([sys.executable, get_abs_path('mcp_server.py')], env=env)
    atexit.register(proc.terminate)

    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f'MCP 服务启动失败, 退出码: {proc.returncode}')
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=0.5):
                return proc, f'http://127.0.0.1:{port}/mcp'
        except OSError:
            time.sleep(0.5)

    proc.terminate()
    raise TimeoutError('MCP 服务启动超时')


def get_mcp_tools() -> list:
    """获取 MCP 工具列表(单例): 首次调用时启动 mcp_server 进程并加载工具,
    后续调用直接返回缓存, 供 ReactAgent / PipelineAgent 共享同一进程.
    双重检查锁: 防止多会话并发首次初始化时重复拉起 MCP 子进程."""
    global _mcp_tools_cache, _mcp_server_proc
    if _mcp_tools_cache is not None:
        return _mcp_tools_cache

    with _mcp_init_lock:
        if _mcp_tools_cache is not None:
            return _mcp_tools_cache

        _mcp_server_proc, url = _start_mcp_server()

        client = MultiServerMCPClient({
            'mediagent': {
                'transport': 'streamable_http',
                'url': url,
            }
        })
        mcp_tools = asyncio.run(client.get_tools())

        def wrap(async_tool):
            def sync_invoke(**kwargs):
                result = asyncio.run_coroutine_threadsafe(
                    async_tool.ainvoke(kwargs), _mcp_loop
                ).result()
                if isinstance(result, list):
                    return ''.join(
                        block.get('text', '')
                        for block in result
                        if isinstance(block, dict) and block.get('type') == 'text'
                    )
                return result

            return StructuredTool.from_function(
                func=sync_invoke,
                name=async_tool.name,
                description=async_tool.description,
                args_schema=async_tool.args_schema,
            )

        _mcp_tools_cache = [wrap(tool) for tool in mcp_tools]
        return _mcp_tools_cache


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=get_mcp_tools(),
            middleware=[monitor_tool, log_before_model, report_prompt_switch,
                        trim_history, medical_guardrail],
            # 内存检查点, 按thread_id保留对话历史, 支持持续多轮对话
            checkpointer=InMemorySaver(),
        )
        # 每个实例(每个用户会话)独立 thread_id, 避免共享 Agent 时对话历史串话
        self._thread_id = str(uuid.uuid4())

    # 工具名称 -> 动作描述, 用于向用户展示Agent当前正在做什么
    TOOL_ACTIONS = {
        'rag_summarize': '检索医学知识库',
        'rag_retrieve': '检索医学知识库原始资料',
        'fetch_patient_history': '查询患者病历',
        'get_patient_vitals': '获取患者生命体征',
        'get_patient_department': '判断就诊科室',
        'get_visit_date': '获取就诊日期',
        'list_patient_ids': '获取患者列表',
    }

    # 工具名称 -> 结果展示标签
    TOOL_RESULT_LABELS = {
        'rag_summarize': '医学知识库检索结果',
        'rag_retrieve': '医学知识库原始资料',
        'fetch_patient_history': '患者病历数据',
        'get_patient_vitals': '患者生命体征',
        'get_patient_department': '就诊科室',
        'get_visit_date': '就诊日期',
        'list_patient_ids': '已建档患者列表',
    }

    def execute_stream(self,query: str):
        """流式执行, 产出结构化事件:
        {'type': 'status',      'text': ...}  Agent当前动作(如正在检索知识库), 会被后续事件覆盖
        {'type': 'tool_result', 'text': ...}  工具检索到的内容, 展示给用户, 会被后续事件覆盖
        {'type': 'content',     'text': ...}  模型输出的正式内容(逐token流式)
        """
        input_dict = {
            'messages':[
                {'role':'user','content':query},
            ]
        }
        reported_tool_calls = set()
        try:
            # recursion_limit限制最大执行步数, 防止工具反复调用导致死循环, 超限后强制退出
            # thread_id标识会话, 每个ReactAgent实例独立, 同一实例内共享对话历史
            config = {'recursion_limit': 25, 'configurable': {'thread_id': self._thread_id}}
            # stream_mode='messages': 逐token推送模型输出, 实现打字机效果的流式输出
            for message, _ in self.agent.stream(input_dict,stream_mode='messages',config=config):

                # 模型token: 逐块输出文本; 检测到工具调用则产生"正在做什么"的状态事件
                # 注意流式块的type是AIMessageChunk, 完整消息是ai, 两者都要处理
                if message.type in ('ai', 'AIMessageChunk'):
                    # 流式块走tool_call_chunks, 完整消息走tool_calls, 两者都兼容
                    tool_calls = getattr(message, 'tool_call_chunks', None) or [
                        {'name': tc['name'], 'id': tc['id']}
                        for tc in getattr(message, 'tool_calls', None) or []
                    ]
                    for tool_call in tool_calls:
                        if tool_call.get('name') and tool_call.get('id') not in reported_tool_calls:
                            reported_tool_calls.add(tool_call['id'])
                            action = self.TOOL_ACTIONS.get(tool_call['name'], f'调用工具{tool_call["name"]}')
                            yield {'type': 'status', 'text': f'⏳ 正在{action}...'}

                    content = message.content
                    # content 可能是内容块列表, 统一提取纯文本
                    if isinstance(content, list):
                        content = ''.join(
                            block.get('text', '')
                            for block in content
                            if isinstance(block, dict) and block.get('type') == 'text'
                        )
                    if content:
                        yield {'type': 'content', 'text': content}
                    continue

                # 工具执行完成 -> 完成状态事件 + 检索内容事件(展示给用户, 后续被整合内容覆盖)
                if message.type == 'tool':
                    action = self.TOOL_ACTIONS.get(message.name, '工具调用')
                    yield {'type': 'status', 'text': f'✅ {action}完成'}

                    result_content = message.content
                    if isinstance(result_content, list):
                        result_content = ''.join(
                            block.get('text', '')
                            for block in result_content
                            if isinstance(block, dict) and block.get('type') == 'text'
                        )
                    if result_content and result_content.strip():
                        label = self.TOOL_RESULT_LABELS.get(message.name, '工具返回')
                        yield {'type': 'tool_result', 'text': f'📄 {label}：\n{result_content.strip()}'}
                    continue
        except GraphRecursionError:
            logger.error('[execute_stream]工具调用次数超出限制, 强制退出')
            yield {'type': 'content', 'text': '工具调用次数超出限制，已强制退出。请调整问题后重试\n'}
        except Exception as e:
            logger.error(f'[execute_stream]生成过程出错: {str(e)}')
            yield {'type': 'content', 'text': f'生成过程出错: {str(e)}，请稍后重试\n'}

if __name__ == '__main__':
    agent = ReactAgent()
    for event in agent.execute_stream('高血压患者在服药期间需要注意什么'):
        print(event['text'],end='',flush=True)
