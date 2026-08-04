import re
from typing import Callable
from langgraph.runtime import Runtime
from langchain.agents import AgentState
from langgraph.types import Command
from langchain.agents.middleware import (
    wrap_tool_call, before_model, after_model, dynamic_prompt, ModelRequest,
)
from langchain_core.messages import ToolMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt.tool_node import ToolCallRequest
from utils.logger_handler import logger
from utils.config_handler import agent_conf
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


# ═══════════════════════════════════════════════════════════════
# 长对话记忆裁剪: 防止多轮对话 + RAG 结果反复注入导致 token 线性膨胀
# ═══════════════════════════════════════════════════════════════

@before_model
def trim_history(state: AgentState, runtime: Runtime):
    """模型调用前裁剪历史: 超出 token 预算时丢弃最早的消息, 保留最近对话.
    以 human 消息为裁剪起点, 保证不会把 tool_call 与其 ToolMessage 拆散."""
    messages = state['messages']
    max_tokens = agent_conf.get('max_history_tokens', 4000)
    if count_tokens_approximately(messages) <= max_tokens:
        return None
    trimmed = trim_messages(
        messages,
        strategy='last',
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on='human',
        include_system=False,
        allow_partial=False,
    )
    if len(trimmed) >= len(messages):
        return None
    logger.info(f'[trim_history]历史消息超出{max_tokens}token预算, {len(messages)}条裁剪为{len(trimmed)}条')
    return {'messages': [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed]}


# ═══════════════════════════════════════════════════════════════
# 医疗输出安全护栏: 检测到具体用药/剂量建议时强制附加免责声明
# ═══════════════════════════════════════════════════════════════

# 具体用药指导特征: 剂量单位 / 给药频次 / 处方措辞
RISK_PATTERN = re.compile(
    r'(\d+\s*(mg|g|ml|毫克|克|毫升|片|粒|单位|IU)|每[日天次]\s*\d|bid|tid|qd|qn|处方|遵医嘱服用|建议服用)'
)

MEDICAL_DISCLAIMER = (
    '\n\n> ⚕️ **安全提示**：以上内容为 AI 辅助生成，仅供临床参考，不构成诊疗建议。'
    '具体诊断与用药请遵从执业医师指导；如出现急重症症状，请立即就医。'
)


@after_model
def medical_guardrail(state: AgentState, runtime: Runtime):
    """输出后置检查: 最终回复(非工具调用)涉及具体用药/剂量时, 强制附加免责声明.
    确定性规则兜底, 不依赖模型自觉遵守 prompt 中的安全约束."""
    messages = state['messages']
    if not messages:
        return None
    last = messages[-1]
    if last.type != 'ai' or getattr(last, 'tool_calls', None):
        return None
    content = last.content
    if isinstance(content, list):
        content = ''.join(
            block.get('text', '') for block in content
            if isinstance(block, dict) and block.get('type') == 'text'
        )
    if not isinstance(content, str) or not content.strip():
        return None
    if not RISK_PATTERN.search(content) or '安全提示' in content:
        return None
    logger.info('[medical_guardrail]检测到具体用药/剂量内容, 附加安全免责声明')
    # add_messages 按 id 覆盖原消息, 实现"修改最后一条AI回复"
    return {'messages': [last.model_copy(update={'content': content + MEDICAL_DISCLAIMER})]}
