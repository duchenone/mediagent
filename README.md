# 智诊通 —— 医疗辅助诊断智能体系统

基于 LangChain + MCP + ChromaDB + 通义千问大模型的医疗辅助诊断智能体系统，支持医学知识检索、患者病历查询、辅助诊断建议和诊断报告生成。

## 功能特性

- **智能医学问答**：基于医学知识库（RAG），提供专业的医学知识检索和问答
- **辅助诊断**：支持获取患者基本信息、生命体征、就诊科室，辅助临床诊断
- **诊断报告生成**：根据患者病历数据、检查结果、知识库参考资料，自动生成结构化诊断报告
- **MCP 工具封装**：6 个专业医疗工具通过 MCP（Model Context Protocol）以独立子进程提供服务
- **多轮对话记忆**：基于 LangGraph Checkpointer 保留对话历史，支持持续追问和上下文引用
- **逐 token 流式输出**：打字机效果，并实时展示 Agent 当前动作（如"正在检索医学知识库"）和检索到的内容
- **手动停止生成**：生成过程中可随时中断
- **提示词确定性切换**：正则匹配用户意图，100% 确定性地切换诊断报告提示词，不依赖 LLM 工具调用
- **容错兜底**：工具调用失败自动降级为错误消息由模型兜底回复；工具调用步数超限强制退出
- **日志监控**：完整的工具调用和模型调用日志，便于问题追踪和审计

## 技术栈

| 类别 | 技术 |
|------|------|
| LLM框架 | LangChain + LangGraph（Checkpointer 对话记忆） |
| 大语言模型 | 通义千问（langchain-qwq / OpenAI 兼容接口，streaming） |
| 工具协议 | MCP（FastMCP + langchain-mcp-adapters，streamable-http 传输） |
| 向量数据库 | ChromaDB |
| 嵌入模型 | text-embedding-v4 |
| Web界面 | Streamlit |
| 文档加载 | PyPDF |
| 配置管理 | PyYAML |
| 环境变量 | python-dotenv |

## 项目结构

```
mediagent/
├── agent/
│   ├── __init__.py
│   ├── react_agent.py          # ReAct Agent 核心类（MCP客户端、流式事件、对话记忆）
│   ├── pipeline/               # 4阶段诊断流水线（病历采集→鉴别诊断→证据检索→报告生成）
│   │   ├── pipeline_agent.py   # PipelineAgent: 组装4个ReAct子图为StateGraph, 流式产出阶段事件
│   │   ├── pipeline_state.py   # PipelineState + SOAP/DDx/Evidence Pydantic 模型
│   │   ├── stage_nodes.py      # 阶段节点: invoke子图→独立parser解析结构化输出→错误降级
│   │   └── tool_sets.py        # 各阶段工具子集(Stage3仅RAG+department过滤)
│   └── tools/
│       ├── __init__.py
│       ├── agent_tools.py      # 医疗工具集（6个工具，全部基于真实数据）
│       └── middleware.py       # 中间件（监控、日志、正则切换提示词）
├── config/
│   ├── agent.yml               # Agent 配置
│   ├── chroma.yml              # ChromaDB 配置
│   ├── prompts.yml             # 提示词路径配置
│   └── rag.yml                 # 模型配置
├── data/                        # 知识库(按专科分子目录,加载时递归扫描并以子目录名作 department 元数据)
│   ├── external/
│   │   └── patient_records.csv # 患者病历数据（患者ID + 就诊日期联合主键）
│   ├── 心血管内科/              # 高血压/冠心病/心衰 (txt + 同名PDF阅读版)
│   ├── 呼吸内科/                # 肺炎/哮喘GINA/COPD
│   ├── 消化内科/                # 消化性溃疡/Hp根除
│   ├── 内分泌科/                # 糖尿病/甲亢
│   ├── 肾内科/                  # 尿路感染/慢性肾病
│   ├── 血液科/                  # 贫血鉴别/缺铁性贫血
│   ├── 骨科/                    # 骨关节炎
│   └── 通用/                    # 疾病诊疗FAQ/诊疗规范/药物手册/急症处理
├── logs/                       # 日志目录
├── model/
│   ├── __init__.py
│   └── factory.py              # 模型工厂（ChatQwen + DashScopeEmbeddings）
├── prompts/
│   ├── main_prompt.txt         # 主系统提示词
│   ├── rag_summarize.txt       # RAG总结提示词（含 {input}/{context} 占位符）
│   └── report_prompt.txt       # 诊断报告提示词
├── rag/
│   ├── __init__.py
│   ├── chroma_db/              # ChromaDB 持久化目录
│   ├── rag_service.py          # RAG 总结服务
│   └── vector_store.py         # 向量存储管理
├── utils/
│   ├── __init__.py
│   ├── config_handler.py       # 配置加载器
│   ├── file_handler.py         # 文件处理工具
│   ├── logger_handler.py       # 日志系统
│   ├── path_tool.py            # 路径工具
│   └── prompt_loader.py        # 提示词加载器
├── .env                        # 环境变量（API Key）
├── .gitignore
├── app.py                      # Streamlit 入口（流式界面、状态展示、手动停止）
├── mcp_server.py               # MCP 服务端（封装6个医疗工具，streamable-http 传输）
├── README.md
└── requirements.txt
```

## 快速开始

### 1. 环境准备

```bash
# 克隆或复制项目后进入目录
cd mediagent

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

编辑 `.env` 文件，填入你的通义千问 API Key：

```
DASHSCOPE_API_KEY=你的API密钥
```

### 3. 初始化知识库

首次运行前，需要将医学知识库文档加载到 ChromaDB：

```bash
python -m rag.vector_store
```

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501` 即可使用。MCP 工具服务会随 Agent 自动启动，无需手动操作。

## 界面交互说明

- **实时状态**：生成过程中实时显示 Agent 当前动作（`⏳ 正在检索医学知识库...` → `✅ 完成`）
- **检索可见**：每次工具查询到的内容（病历、体征、知识库资料）即时展示，随后被整合答案覆盖
- **流式输出**：模型回复逐 token 输出，打字机效果
- **手动停止**：生成中点击「停止生成」可随时中断，保留已生成内容
- **持续追问**：同一浏览器会话内保留完整对话历史，可追问（如"他应该在哪个科室就诊"）

## 配置说明

### chroma.yml

```yaml
collection_name: mediagent          # ChromaDB 集合名称
persist_directory: rag/chroma_db    # 向量持久化路径
k: 3                                # 检索返回文档数
data_path: data                     # 知识库文档目录
allow_knowledge_file_type: ["txt","pdf"]  # 允许的文件类型
chunk_size: 200                     # 文本分片大小
chunk_overlap: 20                   # 分片重叠大小
```

### agent.yml

```yaml
external_data_path: data/external/patient_records.csv  # 患者病历数据路径
```

## 可用工具

6 个医疗工具通过 MCP（Model Context Protocol）封装，由 `mcp_server.py` 提供服务（streamable-http 传输）。`ReactAgent` 启动时自动拉起 MCP 服务子进程（动态分配本地端口）并通过 `langchain-mcp-adapters` 加载工具，无需手动启动；进程退出时自动回收。

所有工具均基于 `patient_records.csv` 真实数据实现，无随机模拟、无空返回：

| 工具 | 功能 | 参数 |
|------|------|------|
| rag_summarize | 医学知识库检索并总结 | query |
| list_patient_ids | 获取所有已建档患者ID列表 | 无 |
| get_patient_vitals | 获取患者最新就诊生命体征 | patient_id |
| get_patient_department | 根据患者最新病情推断就诊科室 | patient_id |
| get_visit_date | 获取患者最新就诊日期 | patient_id |
| fetch_patient_history | 获取患者病历 | patient_id, visit_date（可选，默认最新） |

## 诊断报告生成流程

系统通过**正则匹配用户输入中的诊断报告意图**（如"生成诊断报告"、"出具报告"），在中间件层 100% 确定性地切换至诊断报告提示词，不依赖 LLM 工具调用。匹配最近两条用户消息，保证多轮对话中报告模式连续。

进入报告模式后，Agent 按以下顺序获取数据：

1. 确认患者ID（未指定则调用 `list_patient_ids` 展示列表并询问用户，禁止随机选择）
2. `get_visit_date` → 获取该患者最新就诊日期（用户指定日期则直接使用）
3. `fetch_patient_history` → 获取患者病历数据
4. 按需调用 `get_patient_vitals`、`get_patient_department`、`rag_summarize`

数据齐全后生成结构化诊断报告。

## 容错机制

- **工具失败兜底**：工具调用异常时返回错误消息而非抛出异常，模型感知失败后给出兜底回复，对话不中断
- **强制退出**：`recursion_limit` 限制最大执行步数，防止工具反复调用导致死循环
- **MCP 服务守护**：MCP 子进程启动失败或超时（120秒）时明确报错；进程随主程序退出自动回收

## 安全声明

- 本系统为AI辅助诊断工具，**不能替代医生的专业诊断和判断**
- 所有诊疗决策需由执业医师结合临床实际情况作出
- 药物建议需在医生指导下使用
- 涉及急重症症状时，系统会建议立即就医

## 许可证

MIT License

## 版本

v2.0.0 | 最后更新 2026-07-23

### v2.0.0 更新内容

- 工具全面 MCP 化（FastMCP + streamable-http，子进程自动管理）
- 对话模型切换为 ChatQwen（OpenAI 兼容接口），支持逐 token 流式输出
- 新增多轮对话记忆（LangGraph Checkpointer）
- 界面实时展示 Agent 动作和检索内容，支持手动停止生成
- 提示词切换改为正则匹配，100% 确定性
- 工具全部基于真实病历数据实现，去除随机模拟
- 修复 RAG 总结提示词缺少 `{input}`/`{context}` 占位符导致检索失效的问题
- 新增工具失败兜底、调用步数强制退出等容错机制
