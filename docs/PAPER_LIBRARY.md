# 论文发现与阅读工作台

页面的“发现论文 · 收藏 · 精读对比”提供五项功能：

1. 输入关键词、起止年份、论文类型和来源，查询 arXiv/OpenAlex/Crossref。Crossref 查询 DOI 登记的期刊和会议论文，在 OpenAlex 限额耗尽时仍可提供正式刊会候选；DOI 登记不等同于同行评审证明。年份同时作为来源查询条件和本地复核条件；类型仅根据来源元数据确认。来源未标明的综述或研究论文会列为排除项，不由模型猜测。
2. 通过 arXiv ID、DOI 和规范化标题合并候选；展示来源版本、筛选理由、摘要、原始页面和 PDF 链接。预印本与正式记录同题时提示核对版本。
3. 勾选论文收藏，或批量下载已选的 arXiv HTTPS PDF。下载失败可重试，非 arXiv PDF 提供原始链接供手动上传。下载后需要点击“重建当前索引”。
4. 在已索引 PDF 中选择一篇生成精读卡片。字段包括研究问题、方法、数据集、指标、结果、局限与复现条件；每项必须引用该论文检索片段，否则标为“未报告”。
5. 选择至少两张卡片生成 Markdown 对比表，可下载。表格展示来源页码，并提醒不同实验条件下的性能数值不能直接排名。

候选清单与卡片按用户和会话保存至 `paper_library.json`，不会自动跨会话共享。精读卡片需要已配置模型接口；论文检索、收藏、下载和对比无需模型调用。引用 ID 校验只能证明生成文本引用了本轮证据，不能证明每个自然语言主张在语义上完全正确。

OpenAlex 匿名查询可能遇到 429 限流。页面“论文来源设置”可保存当前用户的 OpenAlex API Key；查询会在短暂限流后重试，额度耗尽时会显示具体提示。Key 按用户保存在本地配置文件，也可通过 `OPENALEX_API_KEY` 环境变量配置。

API 包括 `POST /api/library/search`、`GET /api/library`、`POST /api/library/save`、`POST /api/library/download`、`POST /api/library/card` 和 `GET /api/library/comparison`。接口沿用项目当前的 `user_id` / `thread_id` 会话隔离方式。

## 刊会与等级

论文发现支持刊会名称、期刊/会议/预印本分类、CCF A/B/C、JCR Q1–Q4、中科院 1–4 区筛选。OpenAlex 的正式刊会与 ISSN 用于身份匹配；arXiv 本身不会作为发表期刊，其作者填写的 journal_ref 另标为待核实信息。

`resources/ccf_2026.json` 从浙江工商大学图书馆公开的 CCF 第七版正式 PDF 表格提取，共 681 条，保存来源 URL、PDF SHA256 和每条页码。等级口径固定为 2026 版，不代表论文发表当年的历史等级，也不证明某篇会议论文属于正式长文。Workshop、Findings、Companion 等名称不自动套用主会等级。

JCR 与中科院分区通过“分区数据与来源”导入 CSV，按用户保存。字段为 `journal,issn,scheme,year,category,level,source_url`，其中 ISSN 可选；scheme 为 JCR 或 CAS，category 应注明学科以及中科院大类/小类口径。导入记录标记为用户提供，并展示来源，不自动声称已由系统独立核实。筛选支持指定年份，默认使用每个体系最新已导入年份；多个学科记录分别显示，满足任一学科所选分区即可命中。无对应数据时不会通过分区筛选。

另有 `GET /api/library/rankings`、`POST /api/library/rankings/import` 接口。当前未内置完整 JCR 或中科院授权数据。
