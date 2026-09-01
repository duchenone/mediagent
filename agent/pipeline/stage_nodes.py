"""流水线各阶段节点函数

节点结构:
- validate: 前置患者校验(纯代码, 不调LLM); 未指定患者时 interrupt() 暂停等用户选择
- stage_1/2: invoke ReAct 子图 → 本地提取 ```json 代码块 → Pydantic 校验, 失败才 fallback 独立 parser
- stage_3: 确定性循环(纯代码): 对 DDx 候选并行调用 rag_retrieve, 构建 EvidenceOutput
- stage_4: invoke ReAct 子图直接产出 Markdown 报告
所有节点 try/except 包裹, 失败设 stage_error 使流水线优雅终止
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from agent.pipeline.pipeline_state import (
    PipelineState, DDxOutput, EvidenceOutput, EvidenceItem,
)
from agent.tools.patient_records import list_patient_ids
from utils.config_handler import load_pipeline_config
from utils.logger_handler import logger
from utils.routing import DEPT_KEYWORDS, PATIENT_ID_PATTERN

_pipeline_conf = load_pipeline_config()

# ── 阶段显示名称 ──
STAGE_NAMES = {
    'validate': '患者校验',
    'stage_1': '病历采集',
    'stage_2': '鉴别诊断',
    'stage_3': '证据检索',
    'stage_4': '报告生成',
}

# ── 各阶段子图的递归上限 (读 pipeline.yml, 独立于父图, 防止单阶段工具调用死循环耗尽父图预算) ──
STAGE_RECURSION_LIMITS = _pipeline_conf.get('stage_recursion_limits') or {
    'stage_1': 12,
    'stage_2': 12,
    'stage_4': 8,
}

# ── 结构化解析失败时是否 fallback 独立 parser (pipeline.yml 开关) ──
STRUCTURED_OUTPUT_FALLBACK = _pipeline_conf.get('structured_output_fallback', True)


def _stage_config(config: RunnableConfig, stage: str) -> RunnableConfig:
    """基于父图 config 生成阶段子图 config: 覆盖递归上限, 替换 thread_id 避免与父图检查点冲突"""
    stage_cfg = dict(config)
    stage_cfg['recursion_limit'] = STAGE_RECURSION_LIMITS.get(stage, 10)
    configurable = dict(stage_cfg.get('configurable') or {})
    configurable['thread_id'] = f"{configurable.get('thread_id', 'pipeline')}_{stage}"
    stage_cfg['configurable'] = configurable
    return stage_cfg


def _last_ai_text(messages: list) -> str:
    """取消息列表中最后一条非空 AI 消息的文本内容"""
    for msg in reversed(messages):
        content = getattr(msg, 'content', '')
        if msg.type in ('ai', 'AIMessageChunk') and isinstance(content, str) and content.strip():
            return content
    return ''


def _extract_json_block(text: str) -> dict | None:
    """从文本中提取 JSON: 优先 ```json 代码块, 其次全文首尾大括号"""
    candidates = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.S)
    for c in reversed(candidates):
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    # 兜底: 取第一个 { 到最后一个 }
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _parse_output(parser, text: str, model_cls, stage_name: str):
    """解析阶段输出: 先本地 JSON 提取(零LLM消耗), 失败才 fallback 独立 parser
    (fallback 受 pipeline.yml 的 structured_output_fallback 开关控制)"""
    data = _extract_json_block(text)
    if data is not None:
        try:
            return model_cls.model_validate(data)
        except Exception as e:
            logger.warning(f'[{stage_name}] 本地JSON校验失败, 尝试parser: {e}')
    if not STRUCTURED_OUTPUT_FALLBACK or parser is None or not text.strip():
        return None
    try:
        return parser.invoke(f'请从以下{stage_name}结果中提取结构化数据:\n\n{text}')
    except Exception as e:
        logger.error(f'[{stage_name}] parser解析也失败: {e}', exc_info=True)
        return None


def _dept_for_diagnosis(diagnosis: str, primary_department: str) -> str:
    """按诊断名关键词确定性分科, 无匹配时用 Stage2 给出的主要专科"""
    for dept, keywords in DEPT_KEYWORDS.items():
        if any(k in diagnosis for k in keywords):
            return dept
    return primary_department or '通用'


# ═══════════════════════════════════════════════════════════════
# 上下文构造
# ═══════════════════════════════════════════════════════════════

def _soap_context(state: PipelineState) -> str:
    return (
        f"【病历采集结果】\n"
        f"患者ID: {state.get('patient_id', '未指定')}\n"
        f"就诊科室: {state.get('department', '未知')}\n"
        f"就诊日期: {state.get('visit_date', '未知')}\n\n"
        f"## 主诉与现病史\n{state.get('soap_subjective', '无')}\n\n"
        f"## 体征与检查结果\n{state.get('soap_objective', '无')}\n\n"
        f"## 初步评估\n{state.get('soap_assessment', '无')}\n\n"
        f"## 建议方向\n{state.get('soap_plan', '无')}\n\n"
        f"请基于以上病历数据生成鉴别诊断（DDx）列表。"
    )


def _report_context(state: PipelineState) -> str:
    return (
        f"【患者病历】\n"
        f"患者ID: {state.get('patient_id', '未指定')}\n"
        f"就诊科室: {state.get('department', '未知')}\n"
        f"就诊日期: {state.get('visit_date', '未知')}\n"
        f"主诉: {state.get('soap_subjective', '无')}\n"
        f"体征与检查: {state.get('soap_objective', '无')}\n"
        f"初步评估: {state.get('soap_assessment', '无')}\n"
        f"建议方向: {state.get('soap_plan', '无')}\n\n"
        f"【鉴别诊断】\n{state.get('ddx_json', '暂无')}\n\n"
        f"【指南证据】\n{state.get('evidence_json', '暂无')}\n\n"
        f"请综合以上数据, 生成一份完整的诊断报告(Markdown格式)。"
    )


# ═══════════════════════════════════════════════════════════════
# 节点
# ═══════════════════════════════════════════════════════════════

def make_validate_node():
    """前置患者校验节点 (纯代码, 零LLM消耗)
    - query 中含患者ID且存在 → 直接写入 state, 跳过 stage_1 的ID确认环节
    - query 中含患者ID但不存在 → stage_error 终止, 不浪费后续阶段 token
    - query 中无患者ID → interrupt() 暂停, 等用户从已建档列表中选择后 resume
    """
    def node(state: PipelineState, config: RunnableConfig) -> dict:
        query = state.get('user_query', '')
        patient_ids = list_patient_ids()

        m = PATIENT_ID_PATTERN.search(query)
        if m:
            pid = m.group(0).upper()
            if pid in patient_ids:
                logger.info(f'[Validate] 患者{pid}校验通过')
                return {
                    'patient_id': pid,
                    'current_stage': 'stage_1',
                    'stage_status': f'✅ 患者 {pid} 校验通过',
                }
            logger.warning(f'[Validate] 患者{pid}不存在, 流水线终止')
            return {
                'current_stage': 'validate',
                'stage_error': f'患者 {pid} 不存在。已建档患者: {"、".join(patient_ids)}',
                'stage_status': f'❌ 患者 {pid} 不存在',
            }

        # 未指定患者 → 暂停, 让用户选择
        logger.info('[Validate] 未指定患者, interrupt 等待用户选择')
        choice = interrupt({
            'type': 'choose_patient',
            'message': '未在问题中检测到患者ID, 请选择要诊断的患者:',
            'options': patient_ids,
        })
        pid = str(choice).strip().upper()
        if pid not in patient_ids:
            return {
                'current_stage': 'validate',
                'stage_error': f'选择的患者 {pid} 不存在',
                'stage_status': f'❌ 患者 {pid} 不存在',
            }
        return {
            'patient_id': pid,
            'current_stage': 'stage_1',
            'stage_status': f'✅ 已选择患者 {pid}',
        }
    return node


def make_stage_1_node(stage_agent, parser):
    """Stage1 病历采集节点"""
    def node(state: PipelineState, config: RunnableConfig) -> dict:
        user_query = state.get('user_query', '')
        pid = state.get('patient_id', '')
        stage_input = {
            'messages': [HumanMessage(content=(
                f"请为患者 {pid} 获取病历数据并整理成SOAP格式:\n\n{user_query}\n\n"
                f"步骤:\n"
                f"1. 调用 fetch_patient_history 获取该患者的详细病历\n"
                f"2. 调用 get_patient_vitals、get_patient_department、get_visit_date 补充信息\n"
                f"3. 以 ```json 代码块输出SOAP结构化结果。未获取到的数据在对应字段填'未获取到'。"
            ))]
        }
        try:
            from agent.pipeline.pipeline_state import SOAPOutput
            result = stage_agent.invoke(stage_input, _stage_config(config, 'stage_1'))
            text = _last_ai_text(result['messages'])
            soap = _parse_output(parser, text, SOAPOutput, '病历采集')
            if soap is None:
                raise ValueError('未能从Stage1输出中解析SOAP结构化数据')

            logger.info(f'[Stage1] 采集完成: patient={soap.patient_id}, dept={soap.department}')
            return {
                'patient_id': soap.patient_id or pid,
                'department': soap.department,
                'visit_date': soap.visit_date,
                'soap_subjective': soap.subjective,
                'soap_objective': soap.objective,
                'soap_assessment': soap.assessment,
                'soap_plan': soap.plan,
                'current_stage': 'stage_2',
                'stage_status': f'✅ 阶段1/4 病历采集完成 — 患者: {soap.patient_id}, 科室: {soap.department}',
            }
        except Exception as e:
            logger.error(f'[Stage1] 失败: {e}', exc_info=True)
            return {
                'current_stage': 'stage_1',
                'stage_error': f'病历采集失败: {str(e)}',
                'stage_status': '❌ 阶段1/4 病历采集失败',
            }
    return node


def make_stage_2_node(stage_agent, parser):
    """Stage2 鉴别诊断节点"""
    def node(state: PipelineState, config: RunnableConfig) -> dict:
        context = _soap_context(state)
        stage_input = {'messages': [HumanMessage(content=context)]}
        try:
            result = stage_agent.invoke(stage_input, _stage_config(config, 'stage_2'))
            text = _last_ai_text(result['messages'])
            ddx = _parse_output(parser, text, DDxOutput, '鉴别诊断')
            if ddx is None:
                raise ValueError('未能从Stage2输出中解析DDx结构化数据')

            ddx_json = ddx.model_dump_json(ensure_ascii=False)
            logger.info(f'[Stage2] 鉴别诊断完成: {len(ddx.candidates)} 个候选, 主要专科={ddx.primary_department}')
            return {
                'ddx_json': ddx_json,
                'primary_department': ddx.primary_department,
                'current_stage': 'stage_3',
                'stage_status': f'✅ 阶段2/4 鉴别诊断完成 — {len(ddx.candidates)} 个候选疾病',
            }
        except Exception as e:
            logger.error(f'[Stage2] 失败: {e}', exc_info=True)
            return {
                'current_stage': 'stage_2',
                'stage_error': f'鉴别诊断失败: {str(e)}',
                'stage_status': '❌ 阶段2/4 鉴别诊断失败',
            }
    return node


def make_stage_3_node(rag_tool):
    """Stage3 证据检索节点 (确定性循环, 无 ReAct agent)
    对每个 DDx 候选: 确定性分科 → 并行调用 rag_retrieve → 代码构建 EvidenceOutput
    rag_tool: MCP 加载的 rag_retrieve 工具 (StructuredTool)
    """
    def node(state: PipelineState, config: RunnableConfig) -> dict:
        try:
            ddx = DDxOutput.model_validate_json(state.get('ddx_json', '{}'))
            primary = state.get('primary_department', '通用')
            candidates = ddx.candidates[:5]  # 最多5个候选

            def retrieve_one(c):
                dept = _dept_for_diagnosis(c.diagnosis, primary)
                query = f'{c.diagnosis} 诊断标准 鉴别 治疗'
                raw = rag_tool.invoke({'query': query, 'department': dept})
                return EvidenceItem(
                    diagnosis=c.diagnosis,
                    department_used=dept,
                    guideline_summary=str(raw),
                )

            # 并行检索 (MCP 持久事件循环支持并发)
            with ThreadPoolExecutor(max_workers=4) as pool:
                bundles = list(pool.map(retrieve_one, candidates))

            evidence = EvidenceOutput(bundles=bundles)
            evidence_json = evidence.model_dump_json(ensure_ascii=False)
            logger.info(f'[Stage3] 证据检索完成: {len(bundles)} 个证据包 (并行)')
            return {
                'evidence_json': evidence_json,
                'current_stage': 'stage_4',
                'stage_status': f'✅ 阶段3/4 证据检索完成 — {len(bundles)} 个证据包',
            }
        except Exception as e:
            logger.error(f'[Stage3] 失败: {e}', exc_info=True)
            return {
                'current_stage': 'stage_3',
                'stage_error': f'证据检索失败: {str(e)}',
                'stage_status': '❌ 阶段3/4 证据检索失败',
            }
    return node


def make_stage_4_node(stage_agent):
    """Stage4 报告生成节点 (报告为 Markdown 文本, 直接使用 agent 输出)"""
    def node(state: PipelineState, config: RunnableConfig) -> dict:
        context = _report_context(state)
        stage_input = {'messages': [HumanMessage(content=context)]}
        try:
            result = stage_agent.invoke(stage_input, _stage_config(config, 'stage_4'))
            final_report = _last_ai_text(result['messages'])
            if not final_report.strip():
                raise ValueError('Stage4未生成报告内容')

            logger.info(f'[Stage4] 报告生成完成: {len(final_report)} 字符')
            return {
                'final_report': final_report,
                'current_stage': 'done',
                'stage_status': '✅ 阶段4/4 报告生成完成',
            }
        except Exception as e:
            logger.error(f'[Stage4] 失败: {e}', exc_info=True)
            return {
                'current_stage': 'stage_4',
                'stage_error': f'报告生成失败: {str(e)}',
                'stage_status': '❌ 阶段4/4 报告生成失败',
            }
    return node
