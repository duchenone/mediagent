# 智诊通 —— 医疗辅助诊断智能体系统

基于 LangChain/LangGraph + MCP + ChromaDB 的医疗辅助诊断系统，支持**双模式**：单 Agent 通用对话 与 4 阶段结构化诊断流水线（病历采集 → 鉴别诊断 → 证据检索 → 报告生成）。

## 功能特性

### 流水线模式（结构化诊断）

- **4 阶段多 Agent 协作**：病历采集 → 鉴别诊断 → 证据检索 → 报告生成，各阶段独立 ReAct 子图，专属提示词 + 工具子集
- **提示词 100% 确定性切换**：阶段路由由 StateGraph 的边硬编码决定，LLM 不参与"进入哪个阶段"的决策，只决定阶段内的工具调用和内容生成
- **患者校验人工节点**：未指定患者 ID 时通过 LangGraph `interrupt()` 暂停，UI 展示有效患者范围（P1001~P1021），用户手动输入 ID 校验后恢复执行
- **中间产物可见**：侧边栏实时展示 SOAP 病历摘要、鉴别诊断表、指南证据包，推理链条透明
- **Stage 3 零 LLM 确定性检索**：对鉴别诊断候选并行调用 RAG（ThreadPoolExecutor），按诊断自动路由到对应专科知识库
- **报告逐 token 流式输出**：Stage 4 报告实时打字机效果

### 单 Agent 模式（通用对话）

- **智能医学问答**：基于医学知识库（RAG），提供专业的医学知识检索和问答
- **辅助诊断**：获取患者基本信息、生命体征、就诊科室，辅助临床判断
- **诊断报告生成**：正则匹配报告意图，确定性切换报告提示词
- **多轮对话记忆**：基于 LangGraph Checkpointer 保留对话历史；`trim_history` 中间件超出 token 预算自动裁剪最早消息，防多轮膨胀
- **输出安全护栏**：`medical_guardrail` 中间件检测到具体用药/剂量内容时，确定性附加免责声明（不依赖模型自觉遵守 prompt）

### 通用能力

- **MCP 工具封装**：7 个专业医疗工具通过 MCP（Model Context Protocol）streamable-http 传输提供服务；子进程动态端口启动，双重检查锁保证多会话下单例安全
- **专科 RAG 过滤**：知识库按 8 个专科组织（子目录名作 department 元数据），检索可按专科过滤
- **混合检索 + 重排序**：BM25（jieba 中文分词）与向量双路召回经 EnsembleRetriever 融合，gte-rerank 精排名次与召回名次二次 RRF 融合，解决单一 rerank 模型对关键词型查询的排序偏差
- **检索质量可度量**：`tests/` 内置 30 条专科评测集（含 10 条同义改写难例），一键对比基线与混合检索命中率
- **双模型提供方**：推理模型支持 Kimi（月之暗面）/ 通义千问切换，嵌入固定用阿里 text-embedding-v4
- **无闪屏流式 UI**：Streamlit fragment 局部刷新，生成过程页面稳定不闪烁
- **手动停止生成**：关闭事件流生成器，真正中断底层 LLM 流（非仅停止显示）
- **容错兜底**：工具失败自动降级、阶段递归限制防死循环、结构化解析本地优先 + LLM parser 兜底、LLM 调用统一超时 + 自动重试
- **日志监控**：完整的工具调用和模型调用日志

## 技术栈

| 类别 | 技术 |
|------|------|
| LLM框架 | LangChain + LangGraph（StateGraph 父子图、interrupt/resume、Checkpointer、中间件体系） |
| 推理模型 | Kimi（ChatOpenAI，sk-kimi- key 自动走 api.kimi.com/coding/v1）/ 通义千问（ChatQwen） |
| 嵌入模型 | 阿里 text-embedding-v4（DashScope） |
| 检索增强 | BM25（rank_bm25 + jieba）混合召回 + gte-rerank 重排（RRF 双序融合） |
| 工具协议 | MCP（FastMCP + langchain-mcp-adapters，streamable-http） |
| 向量数据库 | ChromaDB（department 元数据过滤） |
| Web界面 | Streamlit（fragment 局部刷新） |
| 配置管理 | PyYAML + python-dotenv |

## 项目结构

```
mediagent/
├── agent/
│   ├── react_agent.py          # 单 Agent 核心（MCP客户端、流式事件、对话记忆）
│   ├── pipeline/               # 4阶段诊断流水线
│   │   ├── pipeline_agent.py   # PipelineAgent: 组装3个ReAct子图+确定性节点为StateGraph
│   │   ├── pipeline_state.py   # PipelineState + SOAP/DDx/Evidence Pydantic 模型
│   │   ├── stage_nodes.py      # 阶段节点: validate(interrupt)/stage_1/2/3(纯代码并行RAG)/4
│   │   └── tool_sets.py        # 各阶段工具子集
│   └── tools/
│       ├── agent_tools.py      # 医疗工具集（7个工具，全部基于真实数据）
│       └── middleware.py       # 中间件(工具监控/日志/动态Prompt/记忆裁剪/安全护栏)
├── config/
│   ├── agent.yml               # Agent 配置
│   ├── chroma.yml              # ChromaDB 配置
│   ├── pipeline.yml            # 流水线配置（各阶段递归限制、prompt cache开关）
│   ├── prompts.yml             # 提示词路径配置
│   └── rag.yml                 # 模型配置（chat_provider: kimi/dashscope）
├── data/                        # 知识库(按专科分子目录, 子目录名即 department 元数据)
│   ├── external/
│   │   └── patient_records.csv # 患者病历数据（21位患者 P1001~P1021）
│   ├── 心血管内科/  呼吸内科/  消化内科/  内分泌科/
│   ├── 肾内科/      血液科/    骨科/      通用/
├── model/
│   └── factory.py              # 模型工厂（按 key 前缀自动选择 Kimi 端点）
├── prompts/
│   ├── main_prompt.txt         # 单 Agent 主提示词
│   ├── pipeline_stage1/2/4.txt # 流水线各阶段提示词(Stage3为纯代码节点, 无prompt)
│   ├── rag_summarize.txt       # RAG 总结提示词
│   └── report_prompt.txt       # 诊断报告提示词
├── rag/
│   ├── rag_service.py          # RAG 服务(混合召回 + rerank 的检索编排)
│   ├── reranker.py             # gte-rerank 重排 + 召回/精排 RRF 双序融合
│   └── vector_store.py         # 向量存储 + BM25语料(递归加载、department元数据、md5去重)
├── tests/
│   ├── eval_dataset.json       # 30条专科检索评测集(含10条同义改写难例)
│   └── run_eval.py             # 评测脚本(--compare 对比朴素向量基线)
├── utils/
│   ├── config_handler.py       # 配置加载器
│   ├── prompt_loader.py        # 提示词加载器
│   ├── prompt_cache.py         # Kimi Context Caching（实验性）
│   └── ...                     # 日志/路径/文件工具
├── scripts/
│   └── generate_pdfs.py        # txt → PDF 批量转换（reportlab 中文字体）
├── app.py                      # Streamlit 入口（双模式、fragment局部刷新、interrupt交互）
├── mcp_server.py               # MCP 服务端（7个医疗工具，streamable-http）
└── requirements.txt
```

## 快速开始

### 1. 环境准备

```bash
cd mediagent
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置 API Key

编辑 `.env` 文件：

```
# 推理模型二选一（在 config/rag.yml 中切换 chat_provider）
MOONSHOT_API_KEY=你的Kimi密钥        # chat_provider=kimi（sk-kimi- 开头自动走 coding 端点）
DASHSCOPE_API_KEY=你的通义千问密钥    # chat_provider=dashscope，且嵌入模型必需
```

> 注意：嵌入模型固定使用阿里 text-embedding-v4，无论选哪个推理模型都需要 `DASHSCOPE_API_KEY`。

### 3. 初始化知识库

首次运行前，将医学知识库文档加载到 ChromaDB：

```bash
python -m rag.vector_store
```

支持 `data/` 下按专科分子目录递归加载，同名 txt+PDF 自动去重，md5 增量更新。

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`。MCP 工具服务随 Agent 自动启动，无需手动操作。

## 流水线模式使用

侧边栏切换到「流水线模式」，输入包含患者 ID 的主诉即可：

```
P1001 胸痛3天，请帮我做诊断分析
```

执行流程：

1. **患者校验**（纯代码）：正则提取患者 ID，不存在则报错终止；未指定则 `interrupt()` 暂停，UI 显示有效范围（P1001~P1021），手动输入 ID 后继续
2. **Stage 1 病历采集**：ReAct Agent 调用患者工具，输出 SOAP 结构化病历（```json 本地提取，失败才走 LLM parser）
3. **Stage 2 鉴别诊断**：ReAct Agent 结合 RAG 生成 3-5 个鉴别诊断（含可能性/依据/关键证据）
4. **Stage 3 证据检索**：零 LLM 确定性节点，对每个候选诊断并行检索对应专科知识库
5. **Stage 4 报告生成**：综合前三阶段产出结构化 Markdown 报告，逐 token 流式输出

侧边栏实时显示阶段进度条和中间产物（SOAP 摘要、鉴别诊断表、证据包）。

## 可用工具

7 个医疗工具通过 MCP 封装（`mcp_server.py`，streamable-http 传输），Agent 启动时自动拉起 MCP 服务。所有工具基于真实数据实现，无随机模拟：

| 工具 | 功能 | 参数 |
|------|------|------|
| rag_summarize | 医学知识库检索并 LLM 总结 | query, department（可选） |
| rag_retrieve | 轻量检索，返回带溯源的原文片段 | query, department（可选） |
| list_patient_ids | 获取所有已建档患者ID列表 | 无 |
| get_patient_vitals | 获取患者最新就诊生命体征 | patient_id |
| get_patient_department | 根据患者最新病情推断就诊科室 | patient_id |
| get_visit_date | 获取患者最新就诊日期 | patient_id |
| fetch_patient_history | 获取患者病历 | patient_id, visit_date（可选） |

## 配置说明

### rag.yml

```yaml
chat_provider: kimi                      # 推理模型提供方: kimi / dashscope
kimi_chat_model: kimi-k2-0905-preview    # Kimi 模型
dashscope_chat_model: qwen3-max          # 通义千问模型
embedding_model_name: text-embedding-v4  # 嵌入模型（固定阿里）
llm_timeout: 60                          # LLM 调用超时(秒)
llm_max_retries: 2                       # LLM 调用自动重试次数
```

### pipeline.yml

```yaml
stage_recursion_limits:        # 各阶段最大工具调用轮数（防死循环; stage_3为纯代码节点无需配置）
  stage_1: 12
  stage_2: 12
  stage_4: 8
structured_output_fallback: true   # 结构化解析失败时启用 LLM parser 兜底
enable_prompt_cache: false         # Kimi Context Caching（实验性）
```

### chroma.yml

```yaml
collection_name: mediagent
persist_directory: rag/chroma_db
k: 5                    # 最终返回分片数(重排精选后)
fetch_k: 10             # 召回阶段候选数(BM25与向量各自召回后融合)
enable_hybrid: true     # BM25+向量混合召回开关
enable_rerank: true     # gte-rerank 重排开关(RRF双序融合)
rerank_model: gte-rerank-v2
rerank_threshold: 0.0   # rerank相关性分数阈值, 0表示不过滤
data_path: data
allow_knowledge_file_type: ["txt", "pdf"]
chunk_size: 200
chunk_overlap: 20
```

## 检索质量评测

内置 30 条专科评测集（8 个专科，含 10 条同义改写难例，如"慢阻肺"→COPD 指南、"近三个月平均血糖"→HbA1c）：

```bash
python tests/run_eval.py            # 评测当前配置(混合检索+重排)
python tests/run_eval.py --compare  # 与优化前配置(朴素向量top-3)对比
```

当前结果：**命中率 100%（30/30），优化前基线 93.3%（28/30）**，混合检索+重排平均延迟 0.62s。
报告自动写入 `tests/eval_report.md`。

## 容错机制

- **工具失败兜底**：工具异常返回错误消息而非抛出，模型感知后兜底回复
- **LLM 调用容错**：统一配置超时（60s）与自动重试（2 次），网络抖动不再直接终止流水线
- **阶段递归限制**：每阶段独立 `recursion_limit`（pipeline.yml 可配），超限强制退出
- **结构化解析降级**：```json 本地正则提取 → Pydantic 校验 → LLM parser 兜底（开关可控）→ `stage_error` 终止（绝不跑错阶段）
- **MCP 服务守护**：持久后台事件循环避免连接反复重建；双重检查锁防多会话并发重复拉起子进程；子进程随主程序退出自动回收
- **知识库一致性**：MD5 增量去重 + 文件更新时按来源清理旧分片，避免新旧知识并存

## 安全声明

- 本系统为AI辅助诊断工具，**不能替代医生的专业诊断和判断**
- 所有诊疗决策需由执业医师结合临床实际情况作出
- 药物建议需在医生指导下使用；系统内置输出安全护栏，检测到具体用药/剂量内容时自动附加免责声明
- 涉及急重症症状时，系统会建议立即就医

## 许可证

MIT License

## 版本

v3.1.0 | 最后更新 2026-08-03

### v3.1.0 更新内容

- RAG 检索升级：BM25(jieba) + 向量混合召回，gte-rerank 精排与召回名次 RRF 二次融合
- 新增检索质量评测集（30 条专科用例含同义改写难例）与评测脚本，命中率 93.3% → 100%
- 新增输出安全护栏中间件（用药/剂量内容自动附加免责声明）
- 新增长对话记忆裁剪中间件（trim_messages，防 token 线性膨胀）
- LLM 调用统一超时 + 自动重试
- MCP 工具单例加载加双重检查锁，修复多会话并发初始化竞态
- "停止生成"改为真正关闭事件流（中断底层 LLM 连接）
- 清理死代码/死配置（stage_recursion_limits 与 structured_output_fallback 配置真正生效）
- 知识库更新时按来源清理旧分片，修复 MD5 去重"只增不改"

### v3.0.0 更新内容

- 新增 4 阶段诊断流水线（多 Agent 协作，StateGraph 父子图架构）
- 阶段路由纯代码控制，提示词切换 100% 确定性
- 患者校验 interrupt 人工节点（范围提示 + 手动输入校验）
- 侧边栏中间产物展示（SOAP/鉴别诊断/证据包）
- Stage 3 零 LLM 确定性并行专科检索
- 知识库按 8 专科组织，RAG 支持 department 元数据过滤
- 新增 rag_retrieve 轻量检索工具（带溯源引用）
- 推理模型支持 Kimi（按 key 前缀自动选择端点）
- UI 改用 Streamlit fragment 局部刷新，消除闪屏
- 报告生成逐 token 流式输出

### v2.0.0

- 工具全面 MCP 化（FastMCP + streamable-http）
- 多轮对话记忆（LangGraph Checkpointer）
- 逐 token 流式输出、实时状态展示、手动停止
- 工具全部基于真实病历数据实现

### v1.0.0

- 首个版本：ReAct 单 Agent 架构（LangChain）
- 医学知识库 RAG 问答（ChromaDB + text-embedding）
- 患者病历查询、生命体征、就诊科室等基础工具
- 诊断报告生成（提示词切换）
- Streamlit Web 界面
