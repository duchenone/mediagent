import re
from typing import Callable
from langgraph.runtime import Runtime
from langchain.agents import AgentState
from langgraph.types import Command
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts,load_report_prompts

# 诊断报告意图正则: 直接匹配用户输入, 确保100%确定性切换, 不依赖LLM调用工具
REPORT_PATTERN = re.compile(
    r'(诊断报告|报告生成|生成.{0,6}报告|出具.{0,6}报告|撰写.{0,6}报告|写.{0,6}报告|做.{0,6}报告)'
)


# 工具执行的监控
@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest],ToolMessage | Command]
) -> ToolMessage | Command:
    logger.info(f'[monitor_tool]执行工具: {request.tool_call["name"]}')
    logger.info(f'[monitor_tool]参入参数: {request.tool_call["args"]}')
    try:
        result =  handler(request)
        logger.info(f'[monitor_tool]工具{request.tool_call["name"]}调用成功')
        return result
    except Exception as e:
        logger.error(f'工具{request.tool_call["name"]}调用失败,原因: {str(e)}')
        # 不抛出异常, 将错误作为工具结果返回, 让模型感知失败并给出兜底回复, 避免整个对话卡死
        return ToolMessage(
            content=f'工具{request.tool_call["name"]}调用失败: {str(e)}',
            tool_call_id=request.tool_call['id'],
            name=request.tool_call['name'],
        )

# 在模型执行前输出日志
@before_model
def log_before_model(
        state: AgentState,
        runtime: Runtime,
):
    logger.info(f'[log_before_model]即将调用模型, 带有{len(state['messages'])}条消息.')
    logger.debug(f'[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}')
    return None

# 动态切换提示词, 在每一次生成提示词之前,调用此函数
@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    # 正则匹配最近两条用户消息中的诊断报告意图(检查两条是为保证多轮对话中,
    # 用户先要求生成报告、再补充患者ID时仍保持报告模式), 命中即切换为诊断报告提示词
    human_messages = [
        message for message in request.messages
        if message.type == 'human' and isinstance(message.content, str)
    ]
    is_report = any(REPORT_PATTERN.search(message.content) for message in human_messages[-2:])

    if is_report:
        logger.info('[report_prompt_switch]检测到诊断报告意图, 切换为诊断报告提示词')
        return load_report_prompts()
    return load_system_prompts()
