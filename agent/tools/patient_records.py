"""患者病历 CSV 的统一加载入口 (单一事实来源)

agent_tools 的查询工具与 pipeline 的 validate 节点共用:
- csv.DictReader 正规解析: 字段内含逗号/引号不会错位, 自动跳过表头
- 患者ID统一大写, 进程内只加载一次
"""

import csv
import os

from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

_patient_data: dict = {}


def load_patient_records() -> dict:
    """加载患者病历, 结构: {patient_id: {visit_date: {主诉/体征/检查结果/既往史}}}"""
    global _patient_data
    if _patient_data:
        return _patient_data

    path = get_abs_path(agent_conf['external_data_path'])
    if not os.path.exists(path):
        raise FileNotFoundError(f'外部病历数据文件{path}不存在')

    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            patient_id = (row.get('患者ID') or '').strip().upper()
            visit_date = (row.get('就诊日期') or '').strip()
            if not patient_id or not visit_date:
                continue
            _patient_data.setdefault(patient_id, {})[visit_date] = {
                '主诉': (row.get('主诉') or '').strip(),
                '体征': (row.get('体征') or '').strip(),
                '检查结果': (row.get('检查结果') or '').strip(),
                '既往史': (row.get('既往史') or '').strip(),
            }
    return _patient_data


def get_patient_records(patient_id: str) -> dict | None:
    """按患者ID查询病历, 统一ID格式; 不存在返回 None"""
    return load_patient_records().get(patient_id.strip().upper())


def list_patient_ids() -> list[str]:
    """已建档患者ID列表(升序)"""
    return sorted(load_patient_records().keys())
