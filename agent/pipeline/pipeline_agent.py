"""4 阶段诊断流水线 Agent

StateGraph 结构:
  validate(患者校验, 纯代码+interrupt) → Stage1(病历采集) → Stage2(鉴别诊断)
    → Stage3(证据检索, 确定性并行循环) → Stage4(报告生成) → END

Stage1/2/4 为 create_agent() 编译的 ReAct 子图 (专属 prompt + 工具子集),
Stage3 为纯代码节点 (对 DDx 候选并行 rag_retrieve, 零 LLM 决策消耗)。
结构化输出优先本地 ```json 提取, 失败才 fallback 独立 parser。
"""

import json
import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain.agents import create_agent

from model.factory import chat_model
from agent.pipeline.pipeline_state import (
    PipelineState,
    SOAPOutput, DDxOutput,
)
from agent.pipeline.stage_nodes import (
    make_validate_node,
    make_stage_1_node, make_stage_2_node,
    make_stage_3_node, make_stage_4_node,
    STAGE_NAMES,
)
from agent.pipeline.tool_sets import build_stage_tools, get_tool_by_name
from agent.tools.middleware import monitor_tool, log_before_model, medical_guardrail
from utils.prompt_loader import load_pipeline_stage_prompt
from utils.config_handler import load_pipeline_config, rag_conf
from utils.logger_handler import logger


class PipelineAgent:
    """4 阶段诊断流水线 Agent"""

    def __init__(self):
        pipeline_conf = load_pipeline_config()

        # Kimi Context Caching (实验性): 预注册各阶段 system prompt
        self._prompt_cache_ids = {}
        if pipeline_conf.get('enable_prompt_cache') and rag_conf.get('chat_provider') == 'kimi':
            from utils.prompt_cache import create_prompt_cache
            for i, name in [(1, 'stage_1_history'), (2, 'stage_2_ddx'), (4, 'stage_4_report')]:
                cache_id = create_prompt_cache(
                    name=name,
                    system_prompt=load_pipeline_stage_prompt(i),
                    model=rag_conf.get('kimi_chat_model', 'kimi-k2-0905-preview'),
                )
                if cache_id:
                    self._prompt_cache_ids[name] = cache_id

        # 加载共享 MCP 工具并按阶段过滤 (Stage3 用 rag_retrieve 工具句柄直接调用)
        stage_1_tools, stage_2_tools, stage_4_tools = build_stage_tools()
        self._rag_tool = get_tool_by_name('rag_retrieve')

        # 结构化解析器 (仅作本地JSON提取失败时的 fallback)
        self._soap_parser = chat_model.with_structured_output(SOAPOutput)
        self._ddx_parser = chat_model.with_structured_output(DDxOutput)

        # 构建 3 个 stage agent (ReAct 子图; Stage3 为纯代码节点)
        self._stage_1_agent = self._build_stage_agent(
            prompt=load_pipeline_stage_prompt(1),
            tools=stage_1_tools,
            name='stage_1_history',
        )
        self._stage_2_agent = self._build_stage_agent(
            prompt=load_pipeline_stage_prompt(2),
            tools=stage_2_tools,
            name='stage_2_ddx',
        )
        self._stage_4_agent = self._build_stage_agent(
            prompt=load_pipeline_stage_prompt(4),
            tools=stage_4_tools,
            name='stage_4_report',
        )

        # 组装流水线 StateGraph
        self.graph = self._build_graph()
        logger.info('[PipelineAgent] 流水线初始化完成')

    def _build_stage_agent(self, prompt: str, tools: list, name: str):
        """构建单个 stage 的 ReAct Agent 子图 (纯文本输出, 结构化解析在节点层处理)
        enable_prompt_cache 且 provider=kimi 时, 尝试将 system prompt 注册为上下文缓存,
        并通过中间件在每次模型调用的 messages 开头注入 cache 引用 (失败自动降级)
        报告生成阶段(stage_4)额外挂载输出安全护栏, 涉及用药剂量时强制附加免责声明"""
        middlewares = [monitor_tool, log_before_model]
        if name == 'stage_4_report':
            middlewares.append(medical_guardrail)
        cache_id = self._prompt_cache_ids.get(name)
        if cache_id:
            from langchain.agents.middleware import wrap_model_call
            from utils.prompt_cache import cache_message

            @wrap_model_call
            def inject_cache(request, handler):
                request.messages.insert(0, cache_message(cache_id))
                return handler(request)

            middlewares.append(inject_cache)
        return create_agent(
            model=chat_model,
            system_prompt=prompt,
            tools=tools,
            middleware=middlewares,
            name=name,
            checkpointer=InMemorySaver(),
        )

    def _build_graph(self):
        """构建流水线 StateGraph: validate → stage_1 → stage_2 → stage_3 → stage_4"""
        builder = StateGraph(PipelineState)

        builder.add_node('validate', make_validate_node())
        builder.add_node('stage_1', make_stage_1_node(self._stage_1_agent, self._soap_parser))
        builder.add_node('stage_2', make_stage_2_node(self._stage_2_agent, self._ddx_parser))
        builder.add_node('stage_3', make_stage_3_node(self._rag_tool))
        builder.add_node('stage_4', make_stage_4_node(self._stage_4_agent))

        builder.add_edge(START, 'validate')

        # 阶段间有条件跳转: 如果 stage_error 非空, 直接跳到 END
        def after_validate(state: PipelineState) -> str:
            return END if state.get('stage_error') else 'stage_1'
        def after_stage_1(state: PipelineState) -> str:
            return END if state.get('stage_error') else 'stage_2'
        def after_stage_2(state: PipelineState) -> str:
            return END if state.get('stage_error') else 'stage_3'
        def after_stage_3(state: PipelineState) -> str:
            return END if state.get('stage_error') else 'stage_4'
        def after_stage_4(state: PipelineState) -> str:
            return END

        builder.add_conditional_edges('validate', after_validate)
        builder.add_conditional_edges('stage_1', after_stage_1)
        builder.add_conditional_edges('stage_2', after_stage_2)
        builder.add_conditional_edges('stage_3', after_stage_3)
        builder.add_conditional_edges('stage_4', after_stage_4)

        return builder.compile(checkpointer=InMemorySaver())

    # ═══════════════════════════════════════════════════════════
    # 流式执行
    # ═══════════════════════════════════════════════════════════

    def execute_stream(self, query: str):
        """从问题开始流式执行流水线, 产出 UI 事件 (见 _drive)"""
        input_state = {
            'user_query': query,
            'messages': [],
            'current_stage': 'validate',
            'patient_id': '',
            'department': '',
            'visit_date': '',
            'soap_subjective': '',
            'soap_objective': '',
            'soap_assessment': '',
            'soap_plan': '',
            'ddx_json': '',
            'primary_department': '',
            'evidence_json': '',
            'final_report': '',
            'stage_error': '',
            'stage_status': '',
        }
        thread_id = str(uuid.uuid4())
        config = {
            'configurable': {'thread_id': thread_id},
            'recursion_limit': 60,
        }
        yield {'type': 'init', 'thread_id': thread_id}
        yield from self._drive(input_state, config)

    def resume_stream(self, thread_id: str, value: str):
        """interrupt 后恢复执行 (用户在UI上选择了患者)"""
        config = {
            'configurable': {'thread_id': thread_id},
            'recursion_limit': 60,
        }
        yield from self._drive(Command(resume=value), config)

    def _drive(self, input_or_command, config):
        """驱动图执行并产出 UI 事件:
        {'type': 'stage_enter'|'stage_done'|'stage_status'|'artifact'|'content'|'interrupt'|'error'|'done'}
        """
        current_stage = 'validate'
        report_streamed = False  # Stage4 报告是否已逐token推送过

        def stage_event_enter(stage_key: str):
            if stage_key == 'validate':
                return {'type': 'stage_enter', 'stage': stage_key,
                        'progress': '0/4', 'text': '🔍 患者校验'}
            num = int(stage_key.split('_')[1])
            return {'type': 'stage_enter', 'stage': stage_key,
                    'progress': f'{num}/4', 'text': f'📋 阶段 {num}/4: {STAGE_NAMES[stage_key]}'}

        def stage_event_done(stage_key: str):
            if stage_key == 'validate':
                return {'type': 'stage_done', 'stage': stage_key, 'text': '✅ 患者校验完成'}
            num = int(stage_key.split('_')[1])
            return {'type': 'stage_done', 'stage': stage_key,
                    'text': f'✅ 阶段 {num}/4 {STAGE_NAMES[stage_key]} 完成'}

        def extract_artifact(node_output: dict, stage_key: str):
            """从节点输出中提取中间产物, 供UI展示推理链条"""
            if stage_key == 'stage_1' and node_output.get('soap_subjective'):
                return {
                    'title': 'SOAP 病历摘要',
                    'fields': {
                        '患者ID': node_output.get('patient_id', ''),
                        '就诊科室': node_output.get('department', ''),
                        '就诊日期': node_output.get('visit_date', ''),
                        '主诉与现病史': node_output.get('soap_subjective', ''),
                        '体征与检查': node_output.get('soap_objective', ''),
                        '初步评估': node_output.get('soap_assessment', ''),
                        '建议方向': node_output.get('soap_plan', ''),
                    },
                }
            if stage_key == 'stage_2' and node_output.get('ddx_json'):
                try:
                    ddx = json.loads(node_output['ddx_json'])
                    return {
                        'title': '鉴别诊断列表',
                        'fields': {
                            '主要专科': node_output.get('primary_department', ''),
                            '临床推理': ddx.get('clinical_reasoning', ''),
                        },
                        'table': {
                            'headers': ['诊断', '可能性', '依据', '关键证据'],
                            'rows': [
                                [c.get('diagnosis', ''), c.get('probability', ''),
                                 c.get('rationale', ''), c.get('key_evidence', '')]
                                for c in ddx.get('candidates', [])
                            ],
                        },
                    }
                except (json.JSONDecodeError, TypeError):
                    return None
            if stage_key == 'stage_3' and node_output.get('evidence_json'):
                try:
                    ev = json.loads(node_output['evidence_json'])
                    return {
                        'title': '指南证据包',
                        'table': {
                            'headers': ['诊断', '检索专科'],
                            'rows': [[b.get('diagnosis', ''), b.get('department_used', '')]
                                     for b in ev.get('bundles', [])],
                        },
                    }
                except (json.JSONDecodeError, TypeError):
                    return None
            return None

        try:
            # stream_mode 混合: updates(节点状态更新) + messages(子图内LLM token)
            # subgraphs=True 使子图(各stage agent)事件也向上传播, namespace 标识来源节点
            for ns, mode, chunk in self.graph.stream(
                input_or_command,
                stream_mode=['updates', 'messages'],
                config=config,
                subgraphs=True,
            ):
                # ── messages: 仅转发 Stage4 的报告生成 token (逐字流式) ──
                if mode == 'messages':
                    if ns and str(ns[0]).split(':')[0] == 'stage_4':
                        message_chunk, _metadata = chunk
                        content = getattr(message_chunk, 'content', '')
                        if isinstance(content, list):
                            content = ''.join(
                                block.get('text', '') for block in content
                                if isinstance(block, dict) and block.get('type') == 'text'
                            )
                        if content and getattr(message_chunk, 'type', '') in ('AIMessageChunk', 'ai'):
                            report_streamed = True
                            yield {'type': 'content', 'text': content}
                    continue

                # ── updates: 只处理父图(namespace为空)的节点更新 ──
                if ns:
                    continue

                node_name = list(chunk.keys())[0]
                node_output = chunk[node_name]

                # interrupt: 等待用户选择患者
                if node_name == '__interrupt__':
                    interrupt_value = node_output[0].value if node_output else {}
                    yield {
                        'type': 'interrupt',
                        'stage': 'validate',
                        'data': interrupt_value,
                    }
                    return  # 流在此结束, 等 resume_stream 继续

                # 阶段错误
                if node_output.get('stage_error'):
                    yield {
                        'type': 'error',
                        'stage': current_stage,
                        'text': node_output['stage_error'],
                    }
                    break

                # 中间产物
                artifact = extract_artifact(node_output, node_name)
                if artifact:
                    yield {'type': 'artifact', 'stage': node_name, 'data': artifact}

                # 阶段推进
                new_stage = node_output.get('current_stage', '')
                if new_stage and new_stage != current_stage:
                    yield stage_event_done(current_stage)
                    if new_stage == 'done':
                        # 若报告未流式推送(如子图流式失效), 兜底全量推送
                        final_report = node_output.get('final_report', '')
                        if final_report and not report_streamed:
                            yield {'type': 'content', 'text': final_report}
                        break
                    yield stage_event_enter(new_stage)
                    current_stage = new_stage

                # 阶段状态文本
                if node_output.get('stage_status'):
                    yield {
                        'type': 'stage_status',
                        'stage': current_stage,
                        'text': node_output['stage_status'],
                    }

        except Exception as e:
            logger.error(f'[PipelineAgent] 流水线执行异常: {e}', exc_info=True)
            yield {
                'type': 'error',
                'stage': current_stage,
                'text': f'流水线执行异常: {str(e)}',
            }

        yield {'type': 'done'}


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    pa = PipelineAgent()
    print('=== PipelineAgent 自测: P1001 胸痛 ===')
    for event in pa.execute_stream('P1001 胸痛3天,请帮我做诊断分析'):
        if event['type'] in ('stage_enter', 'stage_done'):
            print(f"\n{event['text']}")
        elif event['type'] == 'artifact':
            print(f"  📦 {event['data']['title']}")
        elif event['type'] == 'stage_status':
            print(f"  {event['text']}")
        elif event['type'] == 'content':
            print(event['text'], end='', flush=True)
        elif event['type'] == 'interrupt':
            print(f"\n⏸️ 等待选择患者: {event['data']}")
        elif event['type'] == 'error':
            print(f"\n❌ {event['text']}")
        elif event['type'] == 'done':
            print('\n=== 流水线结束 ===')
