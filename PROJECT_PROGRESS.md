# 项目进展

## 2026-10-28：Skill + 独立科研 Agent Harness

- 移除 LangGraph 编排与依赖，保留原有 RAG/MCP 能力；独立运行 decide / tool / finish / check 循环。模型选择 Skill 和下一步，代码限制 Skill、工具与参数；完成检查不通过会返回循环。
- 新增仓库受信任 Skill：`paper_discovery`、`evidence_reading`、`research_synthesis`。每个工具必须唯一归属一个 Skill，运行时会拒绝越权组合。
- 工具包括会话论文列表、arXiv/OpenAlex/Crossref 搜索、按来源过滤的证据检索、人工确认入库。确认后继续原始目标并沿用预算；取消或审批过期后可补充说明恢复。
- 新增步数、工具次数、模型次数、累计执行时间和上下文字符限制；失败调用也记账，API返回token usage。时间对原生PDF操作为协作式检查，不是强制取消。
- 模型和工具派发前保存任务快照；原始目标、固定要求、证据、观察、预算和轨迹可跨进程恢复。相同任务用文件锁禁止并发执行。
- 页面新增科研任务模式、预算设置、持久审批、任务轨迹、恢复入口与Markdown下载；API新增任务查询、恢复和报告接口。
- 合成千问联调通过：临时空会话、无真实论文或偏好，2步决策、1次工具、4次模型调用，状态completed。真实私人会话联调曾被自动审批拒绝，随后使用隔离的合成会话完成兼容性验证。
- 修正无PDF证据时虚构unknown页码引用也可通过检查的问题。
- 验证覆盖完整循环、失败重规划、预算、Skill/参数越权拦截、引用检查、跨实例恢复、审批继续、Crossref、API与Streamlit重跑。详细边界见docs/HARNESS.md。

## 2026-10-27：模型加载兼容修复

- 确认 8501 当前运行另一个旧项目目录 `RAG科研论文问答系统`，为本项目另启 8502，未修改旧项目文件。
- Embedding 串行初始化、共享完整模型、固定 CPU 并优先使用本地快照；失败或残留 meta 参数的模型不进入缓存。
- 实测发现 macOS PyTorch 与 FAISS 联合使用时，多线程编码崩溃；单线程配置后真实编码和检索通过。
- 验证：65 项回归测试通过；真实文本编码形状 (2, 384)、meta 参数为 0、FAISS 召回成功；Streamlit AppTest 页面无异常，8502 健康检查成功。

## 2026-10-27：可靠性与 Harness 修正

- 统一页面上传、API 上传和搜索入库的 PDF 存储，按内容哈希去重，兼容导入旧目录；追加论文会保留旧论文与问答历史。
- 向量索引与 Chunk 元数据通过不可变代目录和原子 CURRENT 指针一起发布，兼容旧索引读取。
- 审批新增操作锁、原子领取、失败重试、崩溃恢复、成功结果复用及用户/会话归属校验；完成状态保留 resolved。
- 修复停用长期记忆后仍使用旧偏好，以及澄清路由跨轮残留。
- 引用核验改为精确来源、证据 ID 和页码绑定；无证据时跳过生成。配置模型后核验警告最多触发一次扩大检索及修订，仍不通过则返回资料不足。
- API 持久化会话历史与元数据；SSE 随工作流节点完成发送事件，尚非模型 token 流。
- 更正 Python 最低版本为 3.11，新增 requirements-dev.txt，并明确默认检索参数与历史评测配置的区别。
- 验证：62 项测试通过，含真实生成 PDF 的两轮追加入库、FAISS 保存恢复与证据 ID 核验；pip check 与语法编译通过。模型调用和外部搜索使用模拟，未复跑论文检索 benchmark。测试有 5 条第三方 SWIG 弃用警告。

## 2026-08-31：完整科研智能体扩展

- 扩展 Resolve 与 Clarify 节点，除显式研究偏好和常见指代外，可识别论文身份、任务目标、比较对象、数据范围、评价指标及图表身份等缺失信息，并生成合并式针对性追问。
- 新增 Report Agent，基于本地论文证据生成结构化科研综述。
- 新增 SQLite 长期记忆：按 user_id 隔离，支持开关、TTL、查看和清除，只保存提取后的显式偏好。
- 新增搜索结果 HITL：带确认 ID、30 分钟 TTL、幂等恢复、PDF 文件头校验和确认后入库。
- Verify 升级为引用映射检查与可选 LLM 结构化语义核验，并保留无模型时的确定性降级。
- 新增 FastAPI 与 SSE：对话、流式进度、PDF 上传、审批恢复、线程、论文和用户记忆接口。
- Streamlit 增加长期记忆管理与搜索结果确认入库界面。
- 自动化测试扩展至 43 项，并完成 Streamlit/FastAPI 启动、API烟雾测试、真实PDF问答、结构化澄清、长期记忆、HITL、MCP、Report Agent与标准检索评测链路验证。
- 新增 paper-tools MCP Server，提供 `search_arxiv`、`search_openalex` 与 `download_arxiv_pdf`；Search Agent 和审批下载通过 MCP Client 调用，支持工具发现缓存、Schema 校验、超时与自动重连。

## 已完成

- 增加独立 Harness 状态循环与直接工作流
- 增加 Skill 注册表、固定加载和工具权限校验
- 增加确定性条件路由
- 增加 Search Agent，支持 arXiv 论文发现
- 增加 Ingest Agent，复用现有PDF解析和索引构建能力
- 增加 Analysis Agent 与 RAG Agent
- 增加 Verify Agent，核对回答引用与检索来源
- 增加原子 JSON 任务快照，按用户/会话/任务保存 Harness 状态
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
