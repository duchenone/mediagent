"""
MCP 服务端: 将医疗工具集封装为 MCP 工具

传输方式由环境变量 MEDIAGENT_MCP_TRANSPORT 控制:
- stdio(默认): 独立调试时使用, python mcp_server.py
- streamable-http: 由 ReactAgent 以子进程方式自动启动并连接
"""
import os

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP

from agent.tools.agent_tools import (
    rag_summarize as _rag_summarize,
    rag_retrieve as _rag_retrieve,
    list_patient_ids as _list_patient_ids,
    get_patient_vitals as _get_patient_vitals,
    get_patient_department as _get_patient_department,
    get_visit_date as _get_visit_date,
    fetch_patient_history as _fetch_patient_history,
)

mcp = FastMCP('mediagent')


@mcp.tool(description='从医学知识库中检索参考资料并生成总结。可选参数 department 按专科过滤检索范围,取值: 心血管内科/呼吸内科/消化内科/内分泌科/肾内科/血液科/骨科/通用; 不指定则全库检索')
def rag_summarize(query: str, department: str = '') -> str:
    return _rag_summarize.invoke({'query': query, 'department': department})


@mcp.tool(description='从医学知识库中检索原始参考片段(不经过LLM总结, 返回原文+真实来源)。可选参数 department 按专科过滤检索范围,取值同 rag_summarize; 适合需要自行分析原始资料的深度推理场景')
def rag_retrieve(query: str, department: str = '') -> str:
    return _rag_retrieve.invoke({'query': query, 'department': department})


@mcp.tool(description='获取外部系统中所有已建档患者的ID列表，以纯字符串形式返回，可展示给用户选择')
def list_patient_ids() -> str:
    return _list_patient_ids.invoke({})


@mcp.tool(description='获取指定患者最新一次就诊的生命体征数据，以纯字符串形式返回；患者不存在时返回提示信息')
def get_patient_vitals(patient_id: str) -> str:
    return _get_patient_vitals.invoke({'patient_id': patient_id})


@mcp.tool(description='根据指定患者最新一次就诊的病情判断其就诊科室，以纯字符串形式返回；患者不存在时返回提示信息')
def get_patient_department(patient_id: str) -> str:
    return _get_patient_department.invoke({'patient_id': patient_id})


@mcp.tool(description='获取指定患者最新一次就诊日期（月份），以纯字符串形式返回；患者不存在时返回提示信息')
def get_visit_date(patient_id: str) -> str:
    return _get_visit_date.invoke({'patient_id': patient_id})


@mcp.tool(description='从外部系统中获取指定患者的详细病历数据，以纯字符串形式返回。优先按就诊日期精确匹配，未指定日期或该日期无记录时，默认返回该患者最新一次就诊的病历。患者不存在时返回提示信息')
def fetch_patient_history(patient_id: str, visit_date: str = '') -> str:
    return _fetch_patient_history.invoke({'patient_id': patient_id, 'visit_date': visit_date})


if __name__ == '__main__':
    transport = os.environ.get('MEDIAGENT_MCP_TRANSPORT', 'stdio')
    if transport == 'streamable-http':
        mcp.settings.host = os.environ.get('MEDIAGENT_MCP_HOST', '127.0.0.1')
        mcp.settings.port = int(os.environ.get('MEDIAGENT_MCP_PORT', '8765'))
    mcp.run(transport=transport)
