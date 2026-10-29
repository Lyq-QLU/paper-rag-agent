from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    base_url: str
    model: str
    note: str
    docs_url: str | None = None
    console_url: str | None = None


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "openai": ProviderPreset(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        note="适合直接使用 OpenAI 官方接口。",
        docs_url="https://platform.openai.com/docs",
        console_url="https://platform.openai.com/api-keys",
    ),
    "deepseek": ProviderPreset(
        name="DeepSeek",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        note="DeepSeek 官方 OpenAI 兼容接口，也可按平台文档改成 deepseek-reasoner 等模型。",
        docs_url="https://api-docs.deepseek.com/",
        console_url="https://platform.deepseek.com/api_keys",
    ),
    "siliconflow": ProviderPreset(
        name="硅基流动 SiliconFlow",
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-V3.2",
        note="硅基流动 OpenAI 格式接口，可在平台模型列表中替换具体模型名。",
        docs_url="https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions",
        console_url="https://cloud.siliconflow.cn/account/ak",
    ),
    "moonshot": ProviderPreset(
        name="Moonshot",
        base_url="https://api.moonshot.cn/v1",
        model="moonshot-v1-8k",
        note="Moonshot 常见 OpenAI 兼容配置。",
        docs_url="https://platform.moonshot.cn/docs",
        console_url="https://platform.moonshot.cn/console/api-keys",
    ),
    "zhipu": ProviderPreset(
        name="智谱 GLM",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-4-flash",
        note="智谱 GLM 常见 OpenAI 兼容配置。",
        docs_url="https://open.bigmodel.cn/dev/api",
        console_url="https://open.bigmodel.cn/usercenter/apikeys",
    ),
    "xiaomi_mimo": ProviderPreset(
        name="小米 MiMo",
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
        model="mimo-v2.5-pro",
        note="MiMo 官方说明支持 OpenAI 与 Anthropic 兼容 API。V2 系列已下线，建议使用 V2.5 系列模型名。",
        docs_url="https://mimo.mi.com/docs/welcome",
        console_url="https://platform.xiaomimimo.com",
    ),
    "custom": ProviderPreset(
        name="自定义 OpenAI 兼容接口",
        base_url="https://your-provider.example.com/v1",
        model="your-model-name",
        note="适合小米等其他平台。前提是平台提供 OpenAI 兼容 Chat Completions 接口。",
    ),
}


def get_provider_options() -> list[str]:
    return [preset.name for preset in PROVIDER_PRESETS.values()]


def get_preset_by_name(name: str) -> ProviderPreset:
    for preset in PROVIDER_PRESETS.values():
        if preset.name == name:
            return preset
    return PROVIDER_PRESETS["custom"]


def build_secrets_example(preset: ProviderPreset) -> str:
    return "\n".join(
        [
            'OPENAI_API_KEY = "你的 API Key"',
            f'OPENAI_BASE_URL = "{preset.base_url}"',
            f'OPENAI_MODEL = "{preset.model}"',
        ]
    )
