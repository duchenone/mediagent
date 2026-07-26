"""各阶段工具子集定义

从共享 MCP 工具列表中按阶段过滤, 使每个 stage agent 只拿到其职责所需的工具。
Stage3 已改为确定性循环, 直接使用 rag_retrieve 工具句柄, 不再构建 ReAct agent。
"""

from agent.react_agent import get_mcp_tools


def get_tool_by_name(name: str):
    """按名称获取单个 MCP 工具句柄"""
    for t in get_mcp_tools():
        if t.name == name:
            return t
    return None


def build_stage_tools():
    """返回各阶段工具列表: (stage1, stage2, stage4)
    - Stage1 病历采集: 仅患者数据工具
    - Stage2 鉴别诊断: 患者数据工具(核验) + rag_retrieve(原始片段, Agent自行消化)
    - Stage4 报告生成: 仅 rag_retrieve(可选补漏)
    """
    all_tools = get_mcp_tools()

    stage_1 = [t for t in all_tools if t.name not in ('rag_summarize', 'rag_retrieve')]

    stage_2 = list(all_tools)

    stage_4 = [t for t in all_tools if t.name == 'rag_retrieve']

    return stage_1, stage_2, stage_4
