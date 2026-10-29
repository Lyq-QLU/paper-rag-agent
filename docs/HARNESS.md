# 科研 Agent Harness

科研任务由独立、受预算约束的 Harness 直接执行，不依赖 LangGraph，也不依赖模型原生 function-calling 协议。仓库内 Skill 提供任务规则和工具权限；模型输出 JSON 决策，代码使用 Pydantic Schema、Skill 权限和工具白名单三层校验。

## 执行图

```text
请求 → Resolve → Route
                  ├─ direct workflow：上传 / 普通问答 / 引用核验
                  └─ standalone harness_decide ←──────────────────┐
                              ├─ 选择 Skill + Tool → execute ─────┤
                              ├─ finish → synthesis → check ──────┘ 未通过
                              │                         ↓ 通过
                              │                      completed
                              └─ clarify / 审批 / 预算停止 → snapshot
```

模型决策时提供原始目标、首次确定并固定的任务要求、最近历史、已启用的偏好、工具观察、有限证据窗口、剩余预算与可用 Skill。模型必须选择 Skill 和下一步；代码验证工具属于该 Skill 后才执行。工具输出、PDF 内容和历史均作为不可信数据，不赋予它们工具执行权限。

## Skills

| Skill | 允许的工具 | 责任 |
|---|---|---|
| `paper_discovery` | `search_papers`, `request_ingestion` | 查询真实元数据，区分预印本、期刊和会议；需要正文时请求确认入库 |
| `evidence_reading` | `list_papers`, `retrieve_evidence` | 从当前知识库读取真实文件、页码和证据 ID |
| `research_synthesis` | 无外部工具 | 基于已收集证据生成结果，并执行独立完成检查 |

Skill 位于 `skills/<name>/SKILL.md`，由 `src/research_skills.py` 的固定清单加载。运行时不接受论文内容或用户输入创建新 Skill，也不允许 Skill 获得 shell、任意文件或任意 URL 权限。

生成交付结果后，另一次模型调用逐项检查任务要求、原始目标是否完整满足、结论是否有依据。代码同时核对引用的真实文件名、片段 ID、页码；失败返回循环继续补证据。没有 PDF 证据时只能报告工具状态和搜索元数据，不能用虚构论文引用通过检查。语义验收仍依赖模型，不能证明所有事实正确。

## 工具

| 工具 | 参数 | 边界 |
|---|---|---|
| `list_papers` | 无 | 当前会话登记论文与索引来源 |
| `search_papers` | query、provider、max_results、year_from、year_to、venue_kind | arXiv / OpenAlex / Crossref；最多 10 项；仅元数据 |
| `retrieve_evidence` | query、sources、top_k | 当前会话索引；真实文件名过滤；最多 12 块 |
| `request_ingestion` | candidate_ids | 只能选择已搜索候选；必须人工确认；目前仅下载 arXiv PDF |

不提供任意文件读写、shell、任意 URL 下载。科学报告由最终生成节点输出，检查通过后保存为 `tasks/<run_id>.md`。已有 PDF 解析、混合检索及 MCP 网关继续复用。科研任务检索工具当前固定使用 Hybrid，以便稳定执行；普通工作流保留原有检索模式设置。

## 预算

| 字段 | 默认值 | 最大值 |
|---|---:|---:|
| `max_steps` | 10 | 30 |
| `max_tool_calls` | 8 | 25 |
| `max_model_calls` | 14 | 40 |
| `max_seconds` | 180 | 900 |
| `context_chars` | 24000 | 48000 |

工具执行失败和模型调用失败也计入次数。审批后的 PDF 下载计入工具次数；失败重试复用已成功下载的文件。决策 / 交付 / 检查分别限制输出为 1600 / 3000 / 1600 tokens，SDK 自动重试关闭。累计 token 数来自接口返回的 usage；不是美元费用硬上限。

时间预算累计模型与工具执行耗时，不计等待人工确认的时间。本地 PDF 解析等原生操作在节点边界检查时间，不强制杀死已运行操作；网络模型调用超时受剩余时间约束。相同工具及参数执行两次后再重复会被判为无进展，避免无限循环。

上下文使用有界的最近历史、观察摘要与证据窗口；保存最多 60 条证据及 100 条执行摘要。模型调用前检查 prompt 字符预算，超限不会发送请求。此版本不提供任意长度研究任务的无损摘要。

## 使用

页面侧栏打开“科研任务模式”。配置模型后默认开启，仍可以关闭回到普通工作流。没有模型时 API 的 `auto` 模式回退到原工作流；显式 `harness` 模式会要求配置模型。

建议任务：

> 根据当前知识库，分别检索两篇论文的实验数据集、评价指标和对比方法，按表格比较，并给出带证据引用的总结。缺少信息时明确说明。

外部文献任务：

> 搜索 multi-objective vehicle routing 的论文，筛选可获取的候选，确认入库后比较实验设置。

每个任务有稳定 `run_id`。页面历史可展开轨迹、查看工具/模型调用次数；已通过检查的结果可下载 Markdown。页面底部可补充说明并恢复已保存任务。恢复沿用累计次数，提高预算后也不会清零已用量。

## API

`POST /api/chat` 与 `POST /api/chat/stream` 新增：

```json
{
  "message": "比较知识库中论文的实验设置",
  "user_id": "marc",
  "thread_id": "research-demo",
  "execution_mode": "harness",
  "budget": {
    "max_steps": 10,
    "max_tool_calls": 8,
    "max_model_calls": 14,
    "max_seconds": 180,
    "context_chars": 24000
  }
}
```

- `execution_mode`：`auto`（API 默认）、`workflow`、`harness`。
- Python `PaperAgentWorkflow.invoke()` 为兼容原调用，默认仍为 `workflow`；页面显式传入所选模式。
- 返回 `harness`，包括状态、要求清单、usage、completion 和 trace。
- SSE 按节点完成发送进度，最终 answer 事件携带 harness。不是模型 token 流。
- `GET /api/tasks/{run_id}?user_id=…&thread_id=…`：读取持久化任务。
- `POST /api/tasks/{run_id}/resume?user_id=…&thread_id=…`：请求体为 `{"message":"补充说明","budget":{…}}`；budget 可省略。
- `GET /api/tasks/{run_id}/report?user_id=…&thread_id=…`：下载完成结果。
- `POST /api/chat/resume?user_id=…&thread_id=…`：仍使用 action_id、selections，可选 budget；批准入库后自动继续原任务。

## 持久化与边界

Harness 将 `data/users/<user>/sessions/<session>/tasks/<run_id>.json` 作为权威恢复快照，按用户、会话和任务 ID 隔离。模型和工具派发前即记账并原子写入；操作系统文件锁阻止同一任务并发执行。审批和长期偏好继续使用 SQLite，但它们不承担 Harness 编排。

状态包括 `running`、`awaiting_approval`、`needs_input`、`completed`、`budget_exhausted`、`stalled`、`failed`。中断后可从最后保存的任务上下文重新决策；只读工具可能重做。成功任务恢复直接返回结果，不再次调用模型。审批保留自己的操作锁、TTL、进度与幂等结果；取消或过期不会删除原始任务。

用户与会话归属校验继续使用现有 user_id 隔离方式，尚无正式身份认证。首版不提供后台队列、主动通知、美元预算硬限制或跨进程强制取消；不要将模型语义验收当作正式科研事实认证。
