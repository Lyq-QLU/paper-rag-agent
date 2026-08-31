# API 配置说明

本项目调用的是 OpenAI 兼容的 Chat Completions 接口。你可以使用 OpenAI 官方接口，也可以使用兼容 OpenAI 格式的其他模型服务，例如 DeepSeek、硅基流动、Moonshot、智谱等。

## 推荐方式：页面直接填写

启动项目后，在左侧“大模型配置”区域填写：

- API Key
- Base URL
- 模型名

点击“保存 API 配置”后，系统会自动写入 `.streamlit/secrets.toml`。这个文件已经被 `.gitignore` 排除，不会被提交。

## 手动方式：Streamlit 私密配置

复制示例文件：

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

然后编辑 `.streamlit/secrets.toml`：

```toml
OPENAI_API_KEY = "你的 API Key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"
```

`.streamlit/secrets.toml` 已经被 `.gitignore` 排除，不会被提交。

## 方式二：环境变量

```bash
export OPENAI_API_KEY="你的 API Key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
streamlit run app.py
```

## 兼容其他模型服务

只要服务兼容 OpenAI 的 `/chat/completions` 格式，通常只需要改：

```toml
OPENAI_BASE_URL = "你的服务地址/v1"
OPENAI_MODEL = "你的模型名"
```

## 常用服务商示例

### DeepSeek

```toml
OPENAI_API_KEY = "你的 DeepSeek API Key"
OPENAI_BASE_URL = "https://api.deepseek.com"
OPENAI_MODEL = "deepseek-chat"
```

### 硅基流动 SiliconFlow

```toml
OPENAI_API_KEY = "你的 SiliconFlow API Key"
OPENAI_BASE_URL = "https://api.siliconflow.cn/v1"
OPENAI_MODEL = "deepseek-ai/DeepSeek-V3.2"
```

### OpenAI

```toml
OPENAI_API_KEY = "你的 OpenAI API Key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"
```

### 自定义服务商，例如小米

小米 MiMo 官网说明 API 平台提供 OpenAI 与 Anthropic 协议兼容 API。接入时不需要改代码，关键是从 MiMo 开放平台复制三项：

- API Key
- OpenAI 兼容 Base URL
- 当前可用模型名

MiMo V2 系列已于 2026-06-30 下线，模型名建议使用 V2.5 系列，例如：

```toml
OPENAI_API_KEY = "你的 MiMo API Key"
OPENAI_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
OPENAI_MODEL = "mimo-v2.5-pro"
```

如果其他服务商提供 OpenAI 兼容接口，也按同样方式填写：

```toml
OPENAI_API_KEY = "你的 API Key"
OPENAI_BASE_URL = "服务商给你的 OpenAI 兼容地址"
OPENAI_MODEL = "服务商给你的模型名"
```

如果服务商没有 OpenAI 兼容接口，而是自定义鉴权、请求字段或响应字段，则需要单独改 `src/llm.py` 写适配器。

## 页面内检查

启动项目后，左侧栏会显示：

- 当前模型
- 当前接口地址
- API Key 是否已配置
- API 连接测试按钮

如果还没有配置 API Key，系统仍然可以上传论文、解析 PDF、分块、Embedding 和 FAISS 检索，只是回答区会展示检索到的论文片段，而不会调用大模型生成完整答案。

## 官方文档参考

- [OpenAI Docs](https://platform.openai.com/docs)
- [DeepSeek API Docs](https://api-docs.deepseek.com/)
- [SiliconFlow Chat Completions](https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions)
- [Moonshot Docs](https://platform.moonshot.cn/docs)
- [智谱 GLM API](https://open.bigmodel.cn/dev/api)
- [小米 MiMo API 文档](https://mimo.mi.com/docs/welcome)
