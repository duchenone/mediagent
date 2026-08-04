"""PipelineAgent 状态与阶段间结构化输出模型"""

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# Pipeline 共享状态 (StateGraph 的 state_schema)
# ═══════════════════════════════════════════════════════════════

class PipelineState(TypedDict):
    # 累积对话消息 (add_messages 自动合并, 供各阶段子图读写)
    messages: Annotated[list[BaseMessage], add_messages]

    # 原始用户输入
    user_query: str

    # ── Stage1 输出 ──
    patient_id: str
    department: str
    visit_date: str
    soap_subjective: str
    soap_objective: str
    soap_assessment: str
    soap_plan: str

    # ── Stage2 输出 ──
    ddx_json: str           # DDxOutput.model_dump_json()
    primary_department: str  # Stage3 主要检索科室

    # ── Stage3 输出 ──
    evidence_json: str       # EvidenceOutput.model_dump_json()

    # ── Stage4 输出 ──
    final_report: str

    # ── 控制字段 ──
    current_stage: str       # "stage_1" → "stage_2" → "stage_3" → "stage_4" → "done"
    stage_error: str         # 非空表示某阶段失败, 流水线终止
    stage_status: str        # UI 显示用 ("正在采集病历数据...")


# ═══════════════════════════════════════════════════════════════
# 阶段间结构化输出模型 (节点层本地解析 ```json → Pydantic 校验,
# 失败才 fallback chat_model.with_structured_output parser)
# ═══════════════════════════════════════════════════════════════

class SOAPOutput(BaseModel):
    """Stage1 输出: 结构化 SOAP 病历摘要"""
    patient_id: str = Field(description="患者ID, 例如 P1001")
    department: str = Field(description="就诊科室, 例如 心血管内科; 未知填 全科医学科")
    visit_date: str = Field(description="就诊日期YYYY-MM格式; 未知填空字符串")
    subjective: str = Field(description="主诉+现病史")
    objective: str = Field(description="体征+辅助检查结果")
    assessment: str = Field(description="初步评估/问题列表")
    plan: str = Field(description="建议检查方向或处置计划; 未知填空字符串")


class DDxCandidate(BaseModel):
    """单个鉴别诊断条目"""
    diagnosis: str = Field(description="诊断名称(中文)")
    probability: str = Field(description="可能性: 高/中/低")
    rationale: str = Field(description="推理依据,须引用患者具体症状/检查结果")
    key_evidence: str = Field(description="支持或反对的关键证据")


class DDxOutput(BaseModel):
    """Stage2 输出: 鉴别诊断列表"""
    candidates: list[DDxCandidate] = Field(description="按概率排序的鉴别诊断列表,3-5个")
    primary_department: str = Field(description="主要专科(用于下一阶段证据检索),如 心血管内科")
    clinical_reasoning: str = Field(description="临床推理路径简述(1-2句话)")


class EvidenceItem(BaseModel):
    """单个疾病的证据包"""
    diagnosis: str = Field(description="诊断名称")
    department_used: str = Field(description="检索时使用的 department 参数")
    guideline_summary: str = Field(description="从知识库检索到的指南证据摘要")
    diagnostic_criteria: str = Field(description="关键诊断标准")
    recommended_tests: str = Field(description="建议进一步检查")


class EvidenceOutput(BaseModel):
    """Stage3 输出: 所有 DDx 候选的证据包集合"""
    bundles: list[EvidenceItem] = Field(description="每个DDx候选一个证据包")
