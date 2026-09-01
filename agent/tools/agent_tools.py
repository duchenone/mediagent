from langchain_core.tools import tool

from agent.tools.patient_records import (
    get_patient_records, list_patient_ids as _list_ids,
)
from utils.logger_handler import logger
from utils.routing import DEPARTMENT_RULES, DEPARTMENTS

# RAG 服务懒加载单例: 避免 import 本模块即打开 Chroma/构建链 (测试与工具脚本无需重资源)
_rag_service = None


def _get_rag():
    global _rag_service
    if _rag_service is None:
        from rag.rag_service import RagSummarizeService
        _rag_service = RagSummarizeService()
    return _rag_service


@tool(description='从医学知识库中检索参考资料并生成总结。可选参数 department 按专科过滤检索范围,取值: 心血管内科/呼吸内科/消化内科/内分泌科/肾内科/血液科/骨科/通用; 不指定则全库检索')
def rag_summarize(query: str, department: str = '') -> str:
    if department and department not in DEPARTMENTS:
        logger.warning(f'[rag_summarize]非法专科参数: {department},改为全库检索')
        department = ''
    return _get_rag().rag_summarize(query, department or None)


@tool(description='从医学知识库中检索原始参考片段(不经过LLM总结, 返回原文+真实来源)。可选参数 department 按专科过滤检索范围,取值同 rag_summarize; 适合需要自行分析原始资料的深度推理场景')
def rag_retrieve(query: str, department: str = '') -> str:
    if department and department not in DEPARTMENTS:
        logger.warning(f'[rag_retrieve]非法专科参数: {department},改为全库检索')
        department = ''
    return _get_rag().rag_retrieve(query, department or None)


@tool(description="获取外部系统中所有已建档患者的ID列表，以纯字符串形式返回，可展示给用户选择")
def list_patient_ids() -> str:
    return '、'.join(_list_ids())


@tool(description="获取指定患者最新一次就诊的生命体征数据，以纯字符串形式返回；患者不存在时返回提示信息")
def get_patient_vitals(patient_id: str) -> str:
    records = get_patient_records(patient_id)
    if not records:
        logger.warning(f'[get_patient_vitals]未检索到患者: {patient_id}的病历数据')
        return f'患者{patient_id}不存在，未检索到生命体征数据'

    latest_date = max(records.keys())
    return f'患者{patient_id.strip().upper()}最新就诊({latest_date})体征: {records[latest_date]["体征"]}'


@tool(description="根据指定患者最新一次就诊的病情判断其就诊科室，以纯字符串形式返回；患者不存在时返回提示信息")
def get_patient_department(patient_id: str) -> str:
    records = get_patient_records(patient_id)
    if not records:
        logger.warning(f'[get_patient_department]未检索到患者: {patient_id}的病历数据')
        return f'患者{patient_id}不存在，无法判断就诊科室'

    latest = records[max(records.keys())]
    text = latest['主诉'] + latest['体征'] + latest['检查结果'] + latest['既往史']
    for department, keywords in DEPARTMENT_RULES:
        if any(keyword in text for keyword in keywords):
            return department
    return '全科医学科'


@tool(description="获取指定患者最新一次就诊日期（月份），以纯字符串形式返回；患者不存在时返回提示信息")
def get_visit_date(patient_id: str) -> str:
    records = get_patient_records(patient_id)
    if not records:
        logger.warning(f'[get_visit_date]未检索到患者: {patient_id}的病历数据')
        return f'患者{patient_id}不存在，未检索到就诊日期'

    # 就诊日期为YYYY-MM格式, 可直接按字符串比较取最新
    return max(records.keys())


@tool(description="从外部系统中获取指定患者的详细病历数据，以纯字符串形式返回。优先按就诊日期精确匹配，未指定日期或该日期无记录时，默认返回该患者最新一次就诊的病历。患者不存在时返回提示信息")
def fetch_patient_history(patient_id: str, visit_date: str = '') -> str:
    records = get_patient_records(patient_id)
    if not records:
        logger.warning(f'[fetch_patient_history]未能检索到患者: {patient_id}的病历数据')
        return f'患者{patient_id}不存在，未检索到病历数据'

    if visit_date in records:
        actual_date = visit_date
    else:
        # 默认取最新一次就诊日期的病历(就诊日期为YYYY-MM格式, 可直接按字符串比较)
        actual_date = max(records.keys())
        logger.info(f'[fetch_patient_history]患者{patient_id}在{visit_date or "未指定日期"}无记录, 默认使用最新就诊日期: {actual_date}')

    record = records[actual_date]
    return (
        f"患者ID: {patient_id.strip().upper()}\n"
        f"就诊日期: {actual_date}\n"
        f"主诉: {record['主诉']}\n"
        f"体格检查: {record['体征']}\n"
        f"辅助检查: {record['检查结果']}\n"
        f"既往史: {record['既往史']}"
    )
