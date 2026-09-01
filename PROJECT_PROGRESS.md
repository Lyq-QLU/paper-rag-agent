# 项目进展

## 2026-08-31：完整科研智能体扩展

- 扩展 Resolve 与 Clarify 节点，除显式研究偏好和常见指代外，可识别论文身份、任务目标、比较对象、数据范围、评价指标及图表身份等缺失信息，并生成合并式针对性追问。
- 新增 Report Agent，基于本地论文证据生成结构化科研综述。
- 新增 SQLite 长期记忆：按 user_id 隔离，支持开关、TTL、查看和清除，只保存提取后的显式偏好。
- 新增搜索结果 HITL：带确认 ID、30 分钟 TTL、幂等恢复、PDF 文件头校验和确认后入库。
- Verify 升级为引用映射检查与可选 LLM 结构化语义核验，并保留无模型时的确定性降级。
- 新增 FastAPI 与 SSE：对话、流式进度、PDF 上传、审批恢复、线程、论文和用户记忆接口。
- Streamlit 增加长期记忆管理与搜索结果确认入库界面。
- 自动化测试扩展至 36 项，并完成 Streamlit/FastAPI 启动、API烟雾测试、真实PDF问答、结构化澄清、长期记忆、HITL、MCP与Report Agent链路验证。
- 新增 paper-tools MCP Server，提供 `search_arxiv`、`search_openalex` 与 `download_arxiv_pdf`；Search Agent 和审批下载通过 MCP Client 调用，支持工具发现缓存、Schema 校验、超时与自动重连。

## 已完成

- 增加 LangGraph `StateGraph` 工作流
- 增加 Supervisor 条件路由节点
- 增加 Search Agent，支持 arXiv 论文发现
- 增加 Ingest Agent，复用现有PDF解析和索引构建能力
- 增加 Analysis Agent 与 RAG Agent
- 增加 Verify Agent，核对回答引用与检索来源
- 增加 SQLite Checkpointer，按用户/会话保存 Agent 状态
- Streamlit 问答入口接入 Agent 工作流并展示路由与核验状态
- 增加 Agent 路由、引用核验与完整图执行测试

- 搭建 Streamlit Web 页面
- 支持上传多篇 PDF 论文
- 支持 PDF 文本解析
- 支持按长度切分论文 Chunk
- 使用 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 做中英文 Embedding
- 使用 FAISS 做本地向量检索
- 支持 Top-K 检索
- 支持论文场景 Prompt 拼接
- 支持 OpenAI 兼容接口调用
- 支持无 API Key 时先展示检索片段
- 增加页面内 API 配置状态和连接测试
- 增加 OpenAI、DeepSeek、硅基流动、Moonshot、智谱、小米 MiMo 和自定义接口配置示例
- 增加本地用户名隔离
- 增加会话隔离：不同会话拥有独立上传文件、RAG 索引和问答历史
- 增加 FAISS 索引持久化
- 增加 chunks 元数据持久化
- 增加会话列表和问答历史持久化
- 回答 Prompt 和检索来源中加入论文页码引用
- 优化论文切片：扩大 chunk，保留段落结构
- 增加结构感知切块：跨页识别章节/小节，在章节内按段落、句子递归切分
- Chunk 元数据增加章节标题和起止页码，并保证不跨章节切块
- 增加 PDF 表格检测和 Markdown 转换，表格作为独立 Chunk 入库
- 增加嵌入图片提取、Figure Caption 绑定和检索命中后的原图展示
- 优化检索策略：默认过滤参考文献片段，多论文问题按论文来源均衡返回
- 增加科研分析任务：结构化总结、核心方法、创新点、实验设置、对比算法判断、复现建议、多论文对比
- 增加 BM25 关键词检索
- 增加 FAISS + BM25 混合检索
- 增加每个用户独立 API Key
- 增加检索来源分数展示：向量、BM25、融合、重排
- 增加 Markdown 问答报告保存与下载
- 增加 RAG 检索评估：测试问题、期望关键词、Top-K 来源、命中率
- 增加当前会话论文列表
- 增加会话重命名
- 增加会话删除
- 增加用已保存论文重建当前会话索引
- 增加历史会话总览：同一用户可查看过去创建的全部会话
- 增加跨会话历史问答搜索：可按关键词查找昨天或更早的问答记录并切换回对应会话
- 增加连续问答：同一会话内可像聊天一样持续追问
- 增加会话上下文增强：回答和检索会参考最近几轮问答来理解“它”“上面的方法”等追问
- 优化页面信息架构：主页面聚焦工作台和连续问答，历史会话、论文资料、RAG 评估收纳到辅助面板
- 优化侧边栏：API 示例和文档默认折叠，减少页面干扰
- 按类 Codex 布局重构页面：左上为工作区，中间为会话列表，左下为用户，右侧只保留当前会话
- 增加 `.streamlit/secrets.toml.example` 私密配置模板
- 增加安装、运行和 API 配置文档

## 你现在要做的关键步骤

1. 复制 API 配置模板：

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

2. 打开 `.streamlit/secrets.toml`，填入你的 API Key：

```toml
OPENAI_API_KEY = "你的 API Key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"
```

3. 重启 Streamlit 页面：

```bash
.venv/bin/streamlit run app.py
```

4. 在页面左侧点击“测试 API 连接”。

5. 上传 1-2 篇 PDF 论文，点击“构建论文知识库”。

6. 用这些问题测试：

- 这篇论文的创新点是什么？
- 这篇论文用了什么算法？
- 实验对比了哪些方法？
- 这个算法能不能作为我的对比算法？
- 这篇论文如何复现？

## 下一步建议

- 增加论文摘要、方法、实验、结论的一键分析按钮
- 增加多篇论文对比分析
- 增强论文筛选检索
- 增加 rerank 重排序
- 整理项目截图和简历描述

更完整的精进路线见 [ROADMAP.md](ROADMAP.md)。
