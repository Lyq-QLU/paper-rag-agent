# Paper RAG Agent 项目面试问答

## 30 秒项目介绍

这是一个面向科研阅读的 LangGraph 多智能体系统。我把论文搜索、PDF 解析、知识库构建、证据问答、报告生成和引用核验拆成多个状态节点，并用条件边完成路由。外部能力通过 MCP Server 暴露 arXiv 搜索和受控下载工具；检索侧采用 Sentence-Transformer + FAISS 的语义召回与 BM25 关键词召回。系统同时提供 Streamlit 页面和 FastAPI/SSE 接口，并用 SQLite 实现工作流恢复、用户长期偏好与人工审批。

## 关键参数（必须能直接回答）

| 项目 | 当前实现 |
| --- | --- |
| Embedding 模型 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 向量索引 | FAISS，使用归一化向量进行相似度检索 |
| 稀疏检索 | 自实现 BM25 |
| Chunk 大小 | 1200 字符 |
| Chunk overlap | 200 字符 |
| 默认 Top-K | 页面/同步接口通常为 4，演示问答可设为 5 |
| LLM | OpenAI-compatible API；当前演示为 `deepseek-chat` |
| 工作流 | LangGraph `StateGraph` + 条件边 |
| 状态持久化 | LangGraph SQLite Checkpointer |
| 长期记忆 | 独立 SQLite，按 `user_id` 隔离，默认 TTL 90 天 |
| 后端 | FastAPI；流式响应采用 SSE |
| 前端 | Streamlit |
| MCP | 官方 Python SDK 1.x，stdio transport，3 个工具 |
| 自动化测试 | 41 项 |

## 高频问题

### 1. 为什么使用 LangGraph？

项目不只是一次“检索后调用模型”，还包含搜索、入库、分析、报告、澄清、核验和人工确认。LangGraph 适合显式保存状态、使用条件边路由以及从 Checkpoint 恢复。普通顺序链也能做简单 RAG，但当流程出现分支、中断和恢复时，可维护性会明显下降。

### 2. 这是真正的多 Agent 吗？

这里的 Agent 是职责隔离的工作流节点，不是多个模型随意对话。Resolve 处理偏好与指代，Supervisor 决定路由，Search 搜索论文，Ingest 建库，Analysis/RAG/Report 生成结果，Verify 做末端检查。节点共享受约束的 `AgentState`，状态转移由代码决定。

### 3. Supervisor 为什么用规则而不是 LLM？

当前意图集合有限，而且上传 PDF、搜索、报告等意图有明显信号。确定性路由成本低、延迟小、容易测试，也不会因模型输出格式变化导致流程走错。缺点是对复合意图和隐含表达的泛化有限，后续可引入 LLM 结构化分类，并保留规则兜底。

### 4. 为什么同时使用向量检索和 BM25？

向量检索适合语义近似表达，但对论文标题、算法缩写、数字指标和专有名词不一定稳定；BM25 对精确词匹配更可靠。系统分别召回后合并、去重，再根据问题是否关注方法、实验、局限或参考文献进行规则重排，从而兼顾语义召回和关键词命中。

### 5. 为什么选 MiniLM-L12-v2？

它支持多语言，模型规模和推理成本适合本地演示，对中英文论文问答比纯英文模型更稳。选择目标是可部署性与效果的平衡，而不是声称它在所有科研语料上最优。生产环境应在自建标注集上与 BGE-M3、E5 等模型进行 Recall@K、延迟和内存对比。

### 6. Chunk 为什么是 1200、Overlap 为什么是 200？

1200 字符能容纳论文中的完整方法段或实验描述，同时避免上下文过长稀释主题；200 字符用于保留相邻分块的边界语义。更重要的是系统优先按章节、段落和句子切分，参数只是超长文本的上限。当前参数属于工程初值，后续需要用检索评测集做网格对比。

### 7. 结构感知切分做了什么？

解析时保存论文名、页码、章节路径和内容类型；正文尽量不跨章节切分；表格转成 Markdown，大表拆分时重复表头；图片作为独立片段保存图注和附近文本。这使模型回答时可以定位到真实文件和页码，而不是只返回一段无来源文本。

### 8. Rerank 是什么模型？

当前不是 Cross-Encoder，而是轻量的规则重排：依据问题意图、章节类型、内容类型和候选得分调整顺序。优点是无额外模型成本、逻辑可解释；缺点是泛化能力有限。简历和面试中应说“规则重排”，不能把它描述成训练过的 Reranker 模型。

### 9. Verify/AnswerGuard 如何工作？

第一层是确定性引用映射：解析回答中的文件名和页码标记，并与本轮召回来源比对，计算引用覆盖率；系统还支持可选的 LLM 语义一致性判断。它能发现来源名称不匹配或缺少引用，但目前还不是完整的逐 Claim NLI，因此不能宣称完全消除幻觉。

### 10. 长期记忆和会话记忆有什么区别？

会话状态由 LangGraph Checkpointer 按 `user_id:session_id` 保存，用于恢复某次流程；长期记忆保存研究方向和回答偏好，按 `user_id` 隔离并支持 TTL、开关和清除。用户原始论文内容不写入长期偏好。

### 11. HITL 如何实现？

Search Agent 返回论文候选后不会直接下载，而是创建带 TTL 的 pending action。用户确认后通过 resume 接口恢复，下载选中的论文并建库；同一个 action 重复提交会返回已处理结果，避免重复下载和重复入库。

### 12. MCP 如何实现？

项目使用官方 Python SDK 构建 `paper-tools` MCP Server，通过 stdio 暴露 `search_arxiv`、`search_openalex` 和 `download_arxiv_pdf`。LangGraph 节点经 MCP Client 完成初始化、工具发现、Schema 校验和调用；客户端缓存工具 Schema，设置调用超时，并在失败后重新发现工具和重连一次。下载工具只允许 HTTPS ArXiv 域名，并检查 50 MB 上限和 PDF 文件头。Checkpoint不属于MCP工具，它负责工作流状态持久化与恢复。

### 13. 为什么使用 SSE？

工作流会经历 resolve、supervisor、rag、verify 等多个阶段。SSE 可以通过一个 HTTP 连接持续返回状态、进度和最终答案，浏览器实现简单，适合服务端单向推送。若需要双向实时控制或高频交互，再考虑 WebSocket。

### 14. 41 项测试覆盖什么？

覆盖 Supervisor 路由、六类结构化缺失信息澄清、MCP Atom 解析、下载域名安全校验、工具 Schema 校验、Search Agent MCP 调用、RAG→Verify 图执行、引用归一化、长期记忆用户隔离、审批幂等、FastAPI、PDF 表格/图片处理与结构化切分等。测试数量不能代替真实效果评测，所以项目另留检索评测入口。

### 15. 项目最大的不足是什么？

目前主要不足是规则路由和规则重排泛化有限；MCP 目前只有 arXiv 两个工具且使用本地 stdio，尚未接入 OpenAlex 和 Streamable HTTP；SQLite 和本地文件适合单机演示；没有正式认证；引用核验尚未做到逐主张证据蕴含。

### 16. 如果要上线，怎么改？

将用户认证与 `user_id` 绑定；对象存储保存 PDF，PostgreSQL 保存业务数据，Redis/任务队列处理异步入库；向量库换成支持过滤和并发的服务；增加限流、审计、指标监控和模型调用重试；建立真实标注集评估 Recall@K、MRR、引用正确率、回答忠实度和端到端延迟。

## 不要说错的边界

- 当前是 **FAISS**，不是 ChromaDB。
- 当前已实现 **arXiv MCP Tools**，但还没有 OpenAlex 和远程 Streamable HTTP。
- 当前 Rerank 是 **规则重排**，不是 Cross-Encoder。
- 41 项是自动化回归测试，不代表 41 个真实业务场景。
- 不应继续使用未经保存和复现的 `Recall@5 = 89%` 作为简历指标。
- Verify 降低引用错误风险，但不能宣称彻底解决幻觉。
