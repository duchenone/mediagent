import os
from utils.config_handler import agent_conf
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

rag = RagSummarizeService()
patient_data = {}

# 科室推断规则: (科室, 关键词列表), 按优先级顺序匹配
DEPARTMENT_RULES = [
    ('内分泌科', ['糖尿', '血糖', '甲状腺', '甲亢', '口干', '多饮多尿', '多尿']),
    ('心血管内科', ['胸痛', '胸闷', '心悸', '冠心病', '心肌梗', '血压', '心绞痛']),
    ('呼吸内科', ['咳嗽', '咳痰', '肺炎', '哮喘', '喘息', '支气管炎', '发热']),
    ('消化内科', ['腹痛', '上腹', '溃疡', '反酸', '嗳气', '十二指肠', '胃镜']),
    ('肾内科', ['肾', '腰痛', '尿频', '尿急', '尿痛', '水肿', '尿常规']),
    ('血液科', ['贫血', '血红蛋白', 'Hb', '铁剂', '乏力', '面色苍白', '出血']),
    ('神经内科', ['头痛', '头晕', '麻木', '抽搐', '癫痫']),
    ('骨科', ['关节', '骨折', '腰椎', '颈椎']),
]


# 知识库专科标签(与 data/ 子目录一致,作为检索元数据 department 的合法取值)
DEPARTMENTS = ['心血管内科', '呼吸内科', '消化内科', '内分泌科', '肾内科', '血液科', '骨科', '通用']

@tool(description='从医学知识库中检索参考资料并生成总结。可选参数 department 按专科过滤检索范围,取值: 心血管内科/呼吸内科/消化内科/内分泌科/肾内科/血液科/骨科/通用; 不指定则全库检索')
def rag_summarize(query: str, department: str = '') -> str:
    if department and department not in DEPARTMENTS:
        logger.warning(f'[rag_summarize]非法专科参数: {department},改为全库检索')
        department = ''
    return rag.rag_summarize(query, department or None)


@tool(description='从医学知识库中检索原始参考片段(不经过LLM总结, 返回原文+真实来源)。可选参数 department 按专科过滤检索范围,取值同 rag_summarize; 适合需要自行分析原始资料的深度推理场景')
def rag_retrieve(query: str, department: str = '') -> str:
    if department and department not in DEPARTMENTS:
        logger.warning(f'[rag_retrieve]非法专科参数: {department},改为全库检索')
        department = ''
    return rag.rag_retrieve(query, department or None)


def load_patient_records():
    """
    从外部CSV文件加载患者病历数据
    {
        "patient_id": {
            "visit_date": {"主诉": xxx, "体征": xxx, ...}
            ...
        },
        ...
    }
    """
    global patient_data
    if not patient_data:
        external_data_path = get_abs_path(agent_conf['external_data_path'])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f'外部病历数据文件{external_data_path}不存在')

        with open(external_data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                patient_id: str = arr[0].replace('"', "")
                chief_complaint: str = arr[1].replace('"', "")
                physical_exam: str = arr[2].replace('"', "")
                lab_results: str = arr[3].replace('"', "")
                past_history: str = arr[4].replace('"', "")
                visit_date: str = arr[5].replace('"', "")

                if patient_id not in patient_data:
                    patient_data[patient_id] = {}

                patient_data[patient_id][visit_date] = {
                    '主诉': chief_complaint,
                    '体征': physical_exam,
                    '检查结果': lab_results,
                    '既往史': past_history,
                }


def _get_patient_records(patient_id: str) -> dict | None:
    """加载并查询患者病历, 统一患者ID格式"""
    load_patient_records()
    return patient_data.get(patient_id.strip().upper())


@tool(description="获取外部系统中所有已建档患者的ID列表，以纯字符串形式返回，可展示给用户选择")
def list_patient_ids() -> str:
    load_patient_records()
    return '、'.join(sorted(patient_data.keys()))


@tool(description="获取指定患者最新一次就诊的生命体征数据，以纯字符串形式返回；患者不存在时返回提示信息")
def get_patient_vitals(patient_id: str) -> str:
    records = _get_patient_records(patient_id)
    if not records:
        logger.warning(f'[get_patient_vitals]未检索到患者: {patient_id}的病历数据')
        return f'患者{patient_id}不存在，未检索到生命体征数据'

    latest_date = max(records.keys())
    return f'患者{patient_id.strip().upper()}最新就诊({latest_date})体征: {records[latest_date]["体征"]}'


@tool(description="根据指定患者最新一次就诊的病情判断其就诊科室，以纯字符串形式返回；患者不存在时返回提示信息")
def get_patient_department(patient_id: str) -> str:
    records = _get_patient_records(patient_id)
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
    records = _get_patient_records(patient_id)
    if not records:
        logger.warning(f'[get_visit_date]未检索到患者: {patient_id}的病历数据')
        return f'患者{patient_id}不存在，未检索到就诊日期'

    # 就诊日期为YYYY-MM格式, 可直接按字符串比较取最新
    return max(records.keys())


@tool(description="从外部系统中获取指定患者的详细病历数据，以纯字符串形式返回。优先按就诊日期精确匹配，未指定日期或该日期无记录时，默认返回该患者最新一次就诊的病历。患者不存在时返回提示信息")
def fetch_patient_history(patient_id: str, visit_date: str = '') -> str:
    records = _get_patient_records(patient_id)
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
