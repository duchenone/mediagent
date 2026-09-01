import threading

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from agent.react_agent import ReactAgent
from agent.pipeline import PipelineAgent
from utils.routing import PIPELINE_ROUTE_PATTERN, PATIENT_ID_PATTERN

st.set_page_config(page_title='智诊通', page_icon='🏥', layout='wide')
st.title('🏥 智诊通 —— 医疗辅助诊断智能体系统')

# 流水线路由: PIPELINE_ROUTE_PATTERN(诊断/报告诉求) 或 PATIENT_ID_PATTERN(患者ID)
# 命中即自动进入4阶段流水线, 确定性规则不经LLM判断; 普通医学问答由单Agent处理
# (正则统一定义在 utils/routing.py)

# ── 侧边栏: 智能路由说明 ──
with st.sidebar:
    st.header('⚙️ 智能路由')
    st.caption('输入 **患者ID**（如 "P1001 头疼三天"）或 **诊断/报告诉求** 自动进入 **4阶段流水线诊断**；'
               '普通医学问答由 **单Agent** 处理。两种模式的执行进度都会实时显示在此处。')
    st.divider()

st.divider()

# ── 初始化 ──
if 'agent' not in st.session_state:
    st.session_state['agent'] = ReactAgent()

if 'pipeline_agent' not in st.session_state:
    st.session_state['pipeline_agent'] = None  # lazy init

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# 渲染历史消息
for message in st.session_state['messages']:
    st.chat_message(message['role']).write(message['content'])

# 用户输入
prompt = st.chat_input(
    placeholder='请输入医学问题（含患者ID+诊断诉求将自动进入流水线诊断，如 P1001 胸痛3天）',
    disabled=bool(st.session_state.get('generating')),
)


# ── 流水线进度 + 中间产物渲染 ──
def render_pipeline_progress(gen: dict):
    stages = [
        ('validate', '🔍 患者校验'),
        ('stage_1', '📋 病历采集'),
        ('stage_2', '🧠 鉴别诊断'),
        ('stage_3', '📚 证据检索'),
        ('stage_4', '📝 报告生成'),
    ]
    current = gen.get('pipeline_stage', 'validate')
    done = set(gen.get('pipeline_stages_done', []))
    error = gen.get('error')

    lines = []
    for key, label in stages:
        if key in done:
            lines.append(f'✅ {label}')
        elif key == current and not error:
            lines.append(f'⏳ **{label}**')
        elif key == current and error:
            lines.append(f'❌ {label} — 出错')
        else:
            lines.append(f'⬜ {label}')
    st.markdown('\n\n'.join(lines))

    for artifact in gen.get('artifacts', []):
        data = artifact['data']
        with st.expander(f"📦 {data['title']}", expanded=False):
            for k, v in data.get('fields', {}).items():
                if v:
                    st.markdown(f'**{k}**: {v}')
            table = data.get('table')
            if table and table.get('rows'):
                header = '| ' + ' | '.join(table['headers']) + ' |'
                sep = '|' + '---|' * len(table['headers'])
                rows = ['| ' + ' | '.join(str(c).replace('\n', ' ')[:60] for c in row) + ' |'
                        for row in table['rows']]
                st.markdown('\n'.join([header, sep] + rows))


def _handle_pipeline_event(event: dict, state: dict):
    """流水线事件 → 共享 gen dict (run/resume 线程共用)"""
    t = event['type']
    if t == 'init':
        state['thread_id'] = event['thread_id']
    elif t == 'stage_enter':
        state['pipeline_stage'] = event['stage']
        state['pipeline_status_text'] = event['text']
        # 阶段进入即在主区域显示"进行中", 避免长LLM调用期间主区域空白
        state['status'] = f"{event['text']}，进行中..."
    elif t == 'stage_done':
        state['pipeline_stages_done'].append(event['stage'])
        state['pipeline_status_text'] = event['text']
    elif t == 'stage_status':
        state['status'] = event['text']
    elif t == 'artifact':
        state['artifacts'].append({'stage': event['stage'], 'data': event['data']})
    elif t == 'content':
        state['chunks'].append(event['text'])
        state['status'] = None
    elif t == 'interrupt':
        state['interrupt'] = event['data']
        state['awaiting_input'] = True
    elif t == 'error':
        state['error'] = event['text']
    return t != 'done'


def _finalize(gen: dict):
    """生成结束: 落盘消息并清理状态"""
    final = ''.join(gen['chunks']).strip()
    if gen['stop']:
        final += '\n\n（已手动停止生成）'
    if gen['error']:
        final += f'\n\n生成出错：{gen["error"]}'
    st.session_state['messages'].append({'role': 'ai', 'content': final if final else '（未生成内容）'})
    del st.session_state['generating']
    del st.session_state['gen']


# ═══════════════════════════════════════════════════════════════
# Fragment: interrupt 患者输入 (静态, 不轮询, 避免输入框失焦)
# ═══════════════════════════════════════════════════════════════
@st.fragment
def interrupt_fragment(gen: dict):
    payload = gen['interrupt']
    options = payload.get('options', [])

    st.chat_message('ai').write(
        f"{payload.get('message', '请输入患者ID:')}\n\n"
        f"有效患者范围: **{options[0]} ~ {options[-1]}**（共 {len(options)} 位）"
        if options else payload.get('message', '请输入患者ID:')
    )

    with st.form(key='patient_form', clear_on_submit=False):
        pid_input = st.text_input('患者ID', placeholder=f'例如 {options[0] if options else "P1001"}')
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button('✅ 确认', type='primary')
        with col2:
            cancelled = st.form_submit_button('❌ 取消')

    if submitted:
        pid = (pid_input or '').strip().upper()
        if pid in options:
            gen['awaiting_input'] = False
            gen['interrupt'] = None
            gen['done'] = False
            pipeline = st.session_state['pipeline_agent']

            def resume_pipeline(state: dict, pipe_agent, thread_id: str, value: str):
                # stream.close() 真正终止生成: 关闭生成器使 GeneratorExit 传播到
                # 底层 LangGraph/HTTP 流, 而非仅仅停止消费事件
                stream = pipe_agent.resume_stream(thread_id, value)
                try:
                    for event in stream:
                        if state['stop']:
                            stream.close()
                            break
                        if not _handle_pipeline_event(event, state):
                            break
                except Exception as e:
                    state['error'] = str(e)
                finally:
                    state['done'] = True

            threading.Thread(target=resume_pipeline,
                             args=(gen, pipeline, gen['thread_id'], pid), daemon=True).start()
            st.rerun(scope='app')
        else:
            st.error(f'输入无效: {pid or "(空)"} 不在已建档患者范围内，请重新输入')
    elif cancelled:
        gen['stop'] = True
        gen['awaiting_input'] = False
        st.rerun(scope='app')


# ═══════════════════════════════════════════════════════════════
# Fragment: 侧边栏进度轮询 (fragment 内不允许 st.sidebar 上下文,
# 故单独抽出, 在 with st.sidebar 中调用; 流水线显示阶段推进, 单Agent显示工具轨迹)
# ═══════════════════════════════════════════════════════════════
@st.fragment(run_every=0.4)
def sidebar_progress_fragment(gen: dict):
    st.divider()
    if gen.get('pipeline_mode'):
        st.subheader('🔄 流水线进度')
        render_pipeline_progress(gen)
    else:
        st.subheader('🤖 Agent 进度')
        steps = gen.get('steps', [])
        st.markdown('\n\n'.join(steps[-8:]) if steps else '⏳ 思考中...')


# ═══════════════════════════════════════════════════════════════
# Fragment: 生成轮询区 (仅此区域 0.4s 局部刷新, 整页不再闪屏)
# ═══════════════════════════════════════════════════════════════
@st.fragment(run_every=0.4)
def generation_fragment(gen: dict):
    ai_box = st.chat_message('ai').empty()
    pipeline_mode = gen.get('pipeline_mode', False)

    if not gen['done']:
        if pipeline_mode:
            display = ''.join(gen['chunks'])
            if gen['status']:
                display += f'\n\n{gen["status"]}'
            if not display:
                display = '⏳ 流水线启动中...'
            elif gen['chunks']:
                display += ' ▌'
            ai_box.write(display)
        else:
            display = ''.join(gen['chunks'])
            if gen['tool_result']:
                display += f"\n\n```text\n{gen['tool_result']}\n```"
            if gen['status']:
                display += f"\n\n{gen['status']}"
            elif display and not gen['tool_result']:
                display += ' ▌'
            ai_box.write(display if display else '⏳ 思考中...')

        if st.button('停止生成'):
            gen['stop'] = True
    else:
        # 生成完成: 落盘并整页刷新一次
        _finalize(gen)
        st.rerun(scope='app')


# ── 用户提交处理 ──
if prompt and not st.session_state.get('generating'):
    st.chat_message('user').write(prompt)
    st.session_state['messages'].append({'role': 'user', 'content': prompt})

    # 确定性意图路由: 含患者ID 或 命中诊断/报告意图 → 流水线; 否则 → 单Agent
    is_pipeline = bool(PIPELINE_ROUTE_PATTERN.search(prompt) or PATIENT_ID_PATTERN.search(prompt))

    gen = {'chunks': [], 'status': None, 'tool_result': None, 'done': False, 'error': None, 'stop': False,
           'pipeline_mode': is_pipeline, 'steps': [],
           'pipeline_stage': None, 'pipeline_stages_done': [], 'pipeline_status_text': '',
           'artifacts': [], 'thread_id': None, 'interrupt': None, 'awaiting_input': False}
    gen['steps'].append('🔀 已自动路由到 **流水线诊断模式**' if is_pipeline
                        else '🔀 已自动路由到 **单Agent问答模式**')
    st.session_state['gen'] = gen
    st.session_state['generating'] = True

    if is_pipeline:
        if st.session_state['pipeline_agent'] is None:
            with st.spinner('正在初始化流水线Agent（首次加载MCP工具）...'):
                st.session_state['pipeline_agent'] = PipelineAgent()
        pipeline = st.session_state['pipeline_agent']

        def run_pipeline(query: str, state: dict, pipe_agent):
            stream = pipe_agent.execute_stream(query)
            try:
                for event in stream:
                    if state['stop']:
                        stream.close()  # 关闭生成器, 真正中断底层 LLM 流
                        break
                    if not _handle_pipeline_event(event, state):
                        break
            except Exception as e:
                state['error'] = str(e)
            finally:
                state['done'] = True

        threading.Thread(target=run_pipeline, args=(prompt, gen, pipeline), daemon=True).start()
    else:
        agent = st.session_state['agent']

        def run_generation(query: str, state: dict, react_agent):
            stream = react_agent.execute_stream(query)
            try:
                for event in stream:
                    if state['stop']:
                        stream.close()  # 关闭生成器, 真正中断底层 LLM 流
                        break
                    if event['type'] == 'status':
                        state['status'] = event['text']
                        state['steps'].append(event['text'])  # 侧边栏进度轨迹
                    elif event['type'] == 'tool_result':
                        state['tool_result'] = event['text']
                        state['status'] = None
                    else:
                        state['chunks'].append(event['text'])
                        state['status'] = None
                        state['tool_result'] = None
            except Exception as e:
                state['error'] = str(e)
            finally:
                state['done'] = True

        threading.Thread(target=run_generation, args=(prompt, gen, agent), daemon=True).start()

    st.rerun()


# ── 生成中: 按状态分发到对应 fragment (模式以提交时的路由结果为准) ──
if st.session_state.get('generating'):
    gen = st.session_state['gen']
    if gen.get('pipeline_mode') and gen.get('awaiting_input') and gen.get('interrupt'):
        interrupt_fragment(gen)
    elif not gen['done']:
        with st.sidebar:
            sidebar_progress_fragment(gen)
        generation_fragment(gen)
    else:
        _finalize(gen)
        st.rerun()
