# 求职版验收清单

## 功能证据

- [x] Streamlit 首页可访问
- [x] FastAPI Swagger 可访问
- [x] DeepSeek API 连接正常
- [x] 真实 PDF 上传并生成索引
- [x] RAG 返回 5 个证据片段
- [x] 回答包含论文来源和页码
- [x] LLM 语义核验可用
- [x] 长期记忆按用户保存研究方向
- [x] arXiv 搜索进入人工确认流程
- [x] 确认后下载并入库
- [x] Report Agent 可生成结构化报告
- [x] 36 项自动化测试通过（含结构化澄清、ArXiv/OpenAlex MCP、Schema 校验与 Agent 调用）

## GitHub 与安全

- [x] README 含架构、启动、测试和限制说明
- [x] `.streamlit/secrets.toml` 已被 `.gitignore` 排除
- [x] `data/users/` 与 Checkpoint 数据已排除
- [x] Git 跟踪文件未发现真实 API Key
- [x] 添加 5 张脱敏演示截图（首页、入库、RAG、记忆、HITL）
- [ ] 添加 2—3 分钟演示视频链接
- [ ] 创建 GitHub Release 或固定演示版本标签

## 投递前人工检查

- [ ] 简历 GitHub 链接可点击
- [ ] 仓库设为公开且从无登录窗口可访问
- [ ] PDF、截图和视频不包含 API Key、私人路径或敏感论文
- [ ] 能在 30 秒内讲完项目介绍
- [ ] 能解释关键参数、技术取舍、失败降级和项目边界
- [ ] 准备一个最难 Bug 及其定位过程
