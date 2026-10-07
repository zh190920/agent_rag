# Fusion RAG Agent

> 融合 **AgentScope / DeepSeek-Harness / DeepAgents / OpenClaw / Claude-Code** 五大主流
> Agent 框架优势，面向企业知识问答场景的**插件化 · 多智能体协作 · 长上下文 · 高精准 ·
> 隐私可控**的生产级 RAG 智能体框架。

核心链路纯标准库实现，**零强依赖即可离线跑通**（EchoLLM + HashEmbedding）；配置真实大模型
与嵌入服务后即得到生成式答案。全链路 `asyncio` 并发，沙箱闸门 + 令牌桶 + 超时预算三重护栏，
单机实测 200 路并发 100% 成功、320+ req/s。

---

## ✨ 五大框架优势映射

| 来源框架 | 吸收的能力 | 本项目落点 |
|---|---|---|
| **AgentScope** | 多智能体分布式编排、多租户隔离、高并发 | `agents/orchestrator.py`、`security/access_control.py`、`Sandbox` 并发闸门 |
| **DeepSeek-Harness** | 插件化微内核、模型无绑定、轨迹回放、沙箱 | `core/kernel.py`、`core/plugin.py`、`llm/router.py`、`observability/tracer.py` |
| **DeepAgents** | 长任务规划拆解、上下文卸载、人工介入 | `planning/decomposer.py`、`planning/context_offload.py`、HITL 短路 |
| **OpenClaw** | 本地化隐私留存、持久化会话、脱敏 | `storage/sqlite_store.py`、`security/redaction.py`、全本地落盘 |
| **Claude-Code** | 长文精读、结果校验、逻辑纠错、**运行时自主调用文件工具 + Skills 渐进披露** | `agents/reasoning.py`、`agents/validator.py`（四维校验 + 回炉纠错）、`agents/tool_agent.py` + `tools/`（function-calling 驱动 agentic 工具循环）、`skills/`（Anthropic Agent Skills 可插拔流程说明书） |

---

## 🏗️ 总体架构

```
用户交互层 (API / CLI / SDK)
        │
多智能体协作调度层  Orchestrator
  ├─ 意图识别 Agent   ├─ 检索 Agent   ├─ 推理 Agent
  ├─ 校验 Agent       ├─ 记忆 Agent   └─ 运维 Agent
        │
RAG 核心业务插件层  混合检索(向量+BM25+RRF+精排) / 任务拆解 / 上下文卸载
        │
微内核底座层  Kernel + PluginRegistry + LLMRouter + Sandbox + CpuExecutor
        │
数据存储与安全层  SQLite(WAL) / LocalVectorStore / 多租户ACL / PII脱敏
        │
可观测运维层（贯穿全层级）  Tracer 轨迹回放 / Metrics / Alerter / FeedbackCollector
```

**请求闭环**：提问 → 限流&租户解析 → 意图识别 → 任务拆解 → 记忆装配 → 并发检索
→ 证据推理 → 校验回炉 → 脱敏 → 组装&持久化 → 事件广播 → 轨迹落盘。

完整设计见 [`DESIGN.md`](./DESIGN.md)。

---

## 📦 目录结构

```
fusion_rag/
├── core/          微内核：kernel / plugin / config / sandbox / events / logging / exceptions
├── llm/           模型无绑定：base（含 function-calling 协议）/ router（重试+降级链）/ echo（离线兜底）/ openai_compatible
├── embedding/     hash（离线）/ openai / sentence_transformers
├── retrieval/     vector_store / bm25（CJK bigram）/ hybrid（RRF融合）/ reranker
├── chunking/      结构感知分块器
├── planning/      decomposer（三级降级拆解）/ context_offload（预算裁剪+磁盘卸载）
├── memory/        short_term（滑窗+滚动摘要）/ long_term / manager
├── tools/         运行时 agentic 文件工具：base / path_jail（越权拦截）/ file_tools（grep·glob·find·list_dir·read_file）/ registry / executor
├── skills/        可插拔技能（Anthropic Agent Skills）：base / frontmatter（YAML 解析）/ loader（分层扫描）/ registry（渐进披露）
├── security/      access_control（租户+KB隔离）/ redaction（PII脱敏）
├── storage/       sqlite_store（WAL，单写多读）/ schema
├── observability/ tracer（span树回放）/ metrics（Prometheus）/ alerter（阈值+冷却）
├── knowledge/     loader / indexer（增量索引）/ feedback（问题沉淀）
├── agents/        intent / retrieval / reasoning / validator / memory_agent / ops / tool_agent（agentic 工具循环）/ orchestrator
├── api/           FastAPI 应用工厂
├── bootstrap.py   全局装配（配置 → 对象图）
├── cli.py         命令行入口
└── types.py       跨模块数据契约
```

---

## 🚀 快速开始

### 环境要求
- **Python ≥ 3.8**（已在 3.8.8 验证通过）
- 核心链路零第三方依赖；可选增强见下。

### 安装
```bash
cd fusion-rag-agent
pip install -e .                 # 核心（离线可跑）
pip install -e ".[api]"          # HTTP 服务：fastapi + uvicorn
pip install -e ".[http]"         # 真异步 HTTP 客户端：aiohttp
pip install -e ".[local-model]"  # 本地嵌入：sentence-transformers
pip install -e ".[yaml]"         # YAML 配置：PyYAML
```

### 一行体验（离线，无需 API Key）
```bash
python -m fusion_rag.cli demo
```
输出：建库 → 提问 → 带引用作答 → 全链路轨迹回放。

---

## 🧭 CLI 用法

```bash
fusion-rag demo                                   # 自包含端到端演示
fusion-rag ask "AgentScope 是什么？" --kb general  # 单轮问答
fusion-rag chat                                    # 交互式多轮会话
fusion-rag index-file ./doc.md --kb general        # 索引单文件
fusion-rag index-dir ./knowledge --globs "**/*.md" # 批量索引目录
fusion-rag stats                                   # 知识库统计
fusion-rag replay <trace_id>                       # 轨迹回放
fusion-rag serve --host 0.0.0.0 --port 8300        # 启动 HTTP 服务
```

未安装为包时可用 `python -m fusion_rag.cli <command>`。公共参数：`--config`、`--data-dir`、`--tenant`。

---

## 🌐 HTTP API

```bash
pip install -e ".[api]"
fusion-rag serve            # 默认 127.0.0.1:8300，交互式文档 /docs
```

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/v1/ask` | 单轮问答（返回答案+引用+置信度+trace_id） |
| `POST` | `/v1/ask/batch` | 批量并发问答 |
| `POST` | `/v1/documents` | 索引一段文本（增量、按哈希去重） |
| `POST` | `/v1/documents/directory` | 批量索引本地目录 |
| `DELETE` | `/v1/documents/{id}` | 删除文档及其索引 |
| `GET` | `/v1/stats` | 知识库统计 |
| `GET` | `/v1/traces/{id}` | 轨迹回放 |
| `GET` | `/v1/feedback` | 热点问题/知识盲区沉淀 |
| `GET` | `/health` | 健康巡检 |
| `GET` | `/metrics` | Prometheus 文本指标 |

异常语义化：限流 `429` / 越权 `403` / 超时 `504` / 需人工 `202` / 参数 `400`。

---

## 🐍 Python SDK

```python
import asyncio
from fusion_rag import Kernel

async def main():
    async with Kernel.from_config_file("config.yaml") as kernel:
        # 建库（增量、幂等）
        await kernel.indexer.add_text(
            "AgentScope 支持多智能体分布式编排与高并发问答。",
            tenant_id="default", kb_id="general", title="agentscope",
        )
        # 问答
        result = await kernel.orchestrator.ask(
            "AgentScope 有什么能力？", tenant_id="default", session_id="s1",
        )
        print(result.answer, result.citations, result.confidence)
        # 批量并发
        results = await kernel.orchestrator.ask_many([
            {"question": "问题一", "tenant_id": "default"},
            {"question": "问题二", "tenant_id": "default"},
        ])

asyncio.run(main())
```

---

## ⚙️ 接入真实模型

复制 `config.example.yaml` 为 `config.yaml`，把 `llm.router.default` 指向真实模型：

```yaml
llm:
  router:
    default: deepseek            # 改这里
    fallback_chain: [deepseek, echo]   # 失败自动降级到离线兜底
  providers:
    deepseek:
      type: openai_compatible    # 兼容 DeepSeek/OpenAI/Ollama/vLLM/通义千问
      base_url: https://api.deepseek.com/v1
      api_key_env: DEEPSEEK_API_KEY
      model: deepseek-chat

embedding:
  provider: sentence_transformers   # 本地嵌入，或改 openai
  sentence_transformers:
    model: BAAI/bge-small-zh-v1.5
```

任何配置项都可用环境变量覆盖：`FUSION_RAG__LLM__ROUTER__DEFAULT=openai`。

---

## 🔒 生产级特性

- **大并发**：全链路 asyncio；CPU 密集（嵌入/BM25/余弦）走独立线程池；SQLite WAL 单写多读；
  全局并发信号量 + 每租户令牌桶 + 请求超时预算三重护栏。
- **韧性降级**：LLM 失败 → 降级链 → 抽取式推理；拆解失败 → 启发式 → 单任务直通；
  嵌入不可用 → HashEmbedding；每步都有兜底，永不空手而归。
- **精准可控**：无证据不作答；`[n]` 内联引用可核对；四维校验（长度/忠实性/引用/信息充分性）
  + 最多 `max_retry` 轮回炉纠错。
- **隐私安全**：知识库/会话/向量/日志全本地留存；入库前 + 输出前双向 PII 脱敏；
  多租户 + 多知识库纵深隔离，越权访问入口即拦截。
- **可观测**：span 树轨迹回放、Prometheus 指标、阈值告警（冷却去重）、热点/盲区反馈沉淀。
- **插件化**：LLM/嵌入/向量库/检索/精排均以 `(kind, name)` 注册，可插拔、可扩展、可热替换。

---

## 🔬 Agentic 文件工具（对标 Claude Code / hermes harness）

除向量检索外，框架支持让**大模型在问答运行时自主多轮调用只读文件工具**，直接探查
授权目录内的原始知识库，读取比向量片段更充分、可核对的原文证据，并给出
`文件名:行号` 引用。整条链路由真实模型的 OpenAI 原生 function-calling 驱动：

```
推理草稿置信度不足 / 证据缺失
  → ToolAgent 循环（≤ max_iters）：
      chat(tools=[grep,glob,list_dir,read_file], tool_choice="auto")
        ├─ 模型请求工具 → PathJail 校验 → 执行 → 以 tool 角色回填结果 → 继续
        └─ 模型直接作答 → 收敛，抽取 `文件:行号` 引用回链到已检索文档
```

- **只读工具集**（纯标准库 `re`/`pathlib`/`os`，零新依赖）：`grep`（正则/字面量全文检索，可带 `context`
  前后文与 `skip` 分页游标）、`glob`（按模式定位文件）、`find`（按名称/大小/修改时间/类型筛选并排序）、
  `list_dir`（列目录结构）、`read_file`（带行号分页精读）。搜索类工具会**跨多个授权 root 合并检索**。
- **PathJail 纵深防御**：所有路径先归一化，拦截 `..` 穿越 / 绝对越界 / 软链逃逸 /
  设备文件（`/dev`、`\\.\`、`nul` 等），越权以错误结果回填给模型而非抛异常。
- **高并发安全**：阻塞 IO 卸载到 `CpuExecutor` 线程池；单调用 `asyncio.wait_for` 超时，
  刻意不复用全局并发闸门以避免嵌套取槽死锁。
- **非破坏 · 能力门控**：仅当首选 provider `supports_tools=True`（真实 OpenAI 兼容模型）
  且证据不足时才触发；离线 `EchoLLM`（`supports_tools=False`）**完全跳过**，现有行为不变。

启用（`config.yaml`）：

```yaml
retrieval:
  agentic_tools: true                 # 打开 agentic 补证开关
  agentic_confidence_threshold: 0.6   # 置信度低于此值才触发
llm:
  router:
    default: deepseek                 # 需指向支持 function-calling 的真实模型
tools:
  enabled: true
  max_iters: 6
  call_timeout: 15
  allow: [grep, glob, find, list_dir, read_file]
  roots:                              # 租户 → 授权目录（支持 ${data_dir} 展开）
    default: "${data_dir}/knowledge"
```

离线体验机制（脚本化 function-calling 替身演示完整循环）：

```bash
python examples/04_agentic_tools.py
```

---

## 🧩 Skills 渐进式披露（Anthropic Agent Skills 模式）

**技能（skill）不是可调用的函数，而是教模型「怎么做某类任务」的流程说明书。** 一个
技能就是一个目录，内含带 YAML frontmatter 的 `SKILL.md`（`name` / `description` /
`allowed-tools` / `metadata` + 正文步骤），可选附带脚本/参考文件。它与上面的**工具**
互补：工具是 function-calling 调用的函数，技能指导模型如何**编排**这些工具。

落地采用 **渐进式披露（progressive disclosure）** 以省 token：

```
启动：扫描 skills.dirs 分层（base→user→project，后层同名覆盖）加载 SKILL.md 元信息
  → ToolAgent system prompt 只注入「name + description + 读取路径」清单
     → 模型判断任务匹配 → 用已有的 read_file 按需读取 SKILL.md 全文 → 照其流程调用工具
```

- **零硬依赖**：frontmatter 优先用 PyYAML，缺失时退化到内置容错解析器（覆盖标量/块/行
  内列表/一层嵌套映射）；名称按规范校验（小写字母/数字/连字符，≤64）。
- **分层来源 + 同名覆盖**：`skills.dirs` 有序列表，后层覆盖前层，便于随部署环境增量定制。
- **统一受控**：技能目录自动并入工具 `PathJail` 授权根，模型 `read_file` 才读得到 `SKILL.md`；
  `read_file` 现**跨所有授权 root** 解析相对路径（与搜索类工具的合并检索一致）。
- **模板变量**：正文支持 `${SKILL_DIR}` / `${SESSION_ID}` / `${TENANT_ID}`（无值原样保留）。
- **非破坏 · 能力门控**：与 agentic 工具同源，仅真实 function-calling 模型驱动时生效；离线
  `EchoLLM` 完全跳过。未配置 `skills.dirs` 时 system prompt 不含技能段，行为不变。

启用（`config.yaml`）：

```yaml
skills:
  enabled: true
  dirs:                             # 分层来源，顺序即优先级（后层同名覆盖前层）
    - "${data_dir}/skills"          # 例如 ${data_dir}/skills/report-writer/SKILL.md
```

离线体验机制（脚本化替身演示「先读技能 → 依流程检索/精读 → 作答」）：

```bash
python examples/05_skills.py
```

---

## 🧪 测试

```bash
pip install -e ".[dev]"
python -m pytest -q          # 96 项：分词/分块/检索/安全/索引/编排/并发/API/agentic 工具/skills
```

测试用 `asyncio.run` 驱动，无需 `pytest-asyncio`，兼容 pytest ≥ 6。

---

## 📄 License

Apache-2.0
