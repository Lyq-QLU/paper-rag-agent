# 基于 LangGraph 与 Hybrid RAG 的科研论文智能体

这是一个面向科研阅读、论文发现和方法复现的状态化智能体系统。项目在原有PDF深度解析与Hybrid RAG基础上，引入LangGraph完成任务路由、节点编排、状态持久化和回答核验。

## Agent 工作流

```text
用户问题 -> Supervisor
              |-- Search Agent   -> arXiv论文发现
              |-- Ingest Agent   -> PDF解析与知识库构建
              |-- Analysis Agent -> 创新点/实验/复现等科研分析
              `-- RAG Agent      -> 本地论文证据检索与回答
                         |
                    Verify Agent -> 引用来源核验
```

- `Supervisor`：根据问题和输入状态执行确定性条件路由，无API Key时也可稳定运行。
- `Search Agent`：调用arXiv Atom API搜索论文元数据。
- `Ingest Agent`：复用PyMuPDF解析、结构感知切分和FAISS/BM25索引构建。
- `Analysis Agent`：处理结构化总结、创新点、实验设置、算法对比和复现建议。
- `RAG Agent`：执行会话上下文增强检索并生成带页码来源的回答。
- `Verify Agent`：检查回答中的来源是否能映射到本轮检索证据。
- `SQLite Checkpointer`：按用户和会话保存LangGraph状态，支持跨轮状态恢复。

核心工作流实现在 `src/agent_workflow.py`，测试位于 `tests/test_agent_workflow.py`。

底层RAG流程：

```text
上传 PDF 论文 -> 提取文本 -> 文本分块 -> Embedding 向量化 -> FAISS 检索 -> 拼接 Prompt -> 大模型回答
```

## 核心功能

- 上传一篇或多篇 PDF 科研论文
- 自动解析 PDF 文本
- 识别论文章节与小节，在章节内按段落/句子递归切分
- 提取 PDF 表格并转换为 Markdown，大表分块时自动重复表头
- 提取嵌入图片，绑定 Figure Caption、页码和附近正文
- 使用 Embedding 生成向量
- 使用 FAISS 建立本地向量索引
- 使用 BM25 与 FAISS 进行混合召回和规则重排
- 使用 LangGraph 编排 Supervisor、Search、Ingest、Analysis、RAG 和 Verify 节点
- 使用 SQLite Checkpointer 持久化 Agent 状态
- 根据问题检索最相关的论文片段
- 调用大模型生成回答
- 支持常用论文分析问题：
  - 这篇论文的创新点是什么？
  - 这篇论文用了什么算法？
  - 实验对比了哪些方法？
  - 这个算法能不能作为我的对比算法？
  - 这篇论文如何复现？

## 项目结构

```text
.
├── app.py                 # Streamlit 页面入口
├── requirements.txt       # Python 依赖
├── src/
│   ├── config.py          # 参数配置
│   ├── llm.py             # 大模型调用
│   ├── paper_loader.py    # PDF 文本解析
│   ├── prompts.py         # Prompt 模板
│   └── rag_pipeline.py    # RAG 核心流程
└── data/
    └── .gitkeep
```

## 安装

建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 配置大模型

如果要调用 OpenAI 兼容接口，请设置环境变量：

```bash
export OPENAI_API_KEY="你的 API Key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
```

如果不设置 `OPENAI_API_KEY`，系统仍然可以运行，会返回“检索到的相关论文片段”，方便先验证 PDF 解析、分块和向量检索流程。

更推荐的本地私密配置方式见 [API_SETUP.md](API_SETUP.md)。

## 运行

```bash
streamlit run app.py
```

打开页面后，上传 PDF 论文，点击“构建论文知识库”，然后输入问题即可。

页面左侧会显示当前模型、接口地址和 API Key 配置状态，也可以直接点击“测试 API 连接”。

## 测试

```bash
python -m pytest -q
```

## 学习重点

短期内先理解这几个概念：

- `Chunk`：把论文切成较短文本块
- `Structure-aware Chunking`：先识别章节边界，再在章节内切块，避免方法、实验和参考文献相互混合
- `content_type`：区分 `text`、`table` 和 `figure`，让图表与普通正文独立检索
- `Embedding`：把文本转换成向量
- `FAISS`：本地向量检索库
- `Top-K Retrieval`：找到最相关的几个文本块
- `Prompt Template`：把问题和检索内容组织成提示词
- `Hallucination`：大模型脱离资料胡说，RAG 用检索内容来约束回答
