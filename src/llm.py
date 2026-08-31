import base64
import mimetypes
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from src.config import get_config


class LLMServiceError(RuntimeError):
    """可以安全展示给页面用户的模型服务异常。"""


def format_api_error(exc: Exception) -> str:
    if isinstance(exc, RateLimitError):
        return "API 请求过于频繁或已达到额度上限，请稍后重试。"
    if isinstance(exc, APIStatusError):
        status = exc.status_code
        if status == 402:
            return "API 余额不足（HTTP 402），请在当前模型服务商控制台充值或更换有余额的 API Key。"
        if status in {401, 403}:
            return "API Key 无效、已过期或没有该模型权限，请检查 Key、Base URL 和模型名。"
        if status == 404:
            return "API 地址或模型名不存在（HTTP 404），请检查 Base URL 和模型名。"
        return f"模型服务返回 HTTP {status}，请检查服务商控制台和 API 配置。"
    if isinstance(exc, APIConnectionError):
        return "无法连接模型 API，请检查网络和 Base URL。"
    return f"模型调用失败：{exc}"


def get_llm_status(user_id: str | None = None) -> dict[str, str | bool]:
    config = get_config(user_id=user_id)
    return {
        "configured": bool(config.openai_api_key),
        "base_url": config.openai_base_url,
        "model": config.openai_model,
    }


def test_llm_connection(user_id: str | None = None) -> str:
    config = get_config(user_id=user_id)
    if not config.openai_api_key:
        return "未配置 OPENAI_API_KEY。"

    client = OpenAI(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
    )
    try:
        response = client.chat.completions.create(
            model=config.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "你只负责测试 API 连通性。",
                },
                {
                    "role": "user",
                    "content": "请只回答：API 连接正常",
                },
            ],
            temperature=0,
            max_tokens=20,
        )
    except (APIStatusError, APIConnectionError, RateLimitError) as exc:
        raise LLMServiceError(format_api_error(exc)) from exc
    content = response.choices[0].message.content or ""
    return content.strip() or "API 连接正常"


def call_llm(
    prompt: str,
    user_id: str | None = None,
    image_paths: list[str] | None = None,
) -> str | None:
    config = get_config(user_id=user_id)
    if not config.openai_api_key:
        return None

    client = OpenAI(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
    )
    user_content = build_user_content(prompt, image_paths or [])
    if isinstance(user_content, list) and is_known_text_only_provider(
        config.openai_base_url,
        config.openai_model,
    ):
        raise LLMServiceError(
            f"当前模型 {config.openai_model} 是纯文本模型，不支持图片输入。"
            "请在页面的模型接口中更换为支持 Vision/多模态输入的 OpenAI 兼容模型。"
        )
    try:
        response = client.chat.completions.create(
            model=config.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个严谨的科研论文阅读助手，只能依据给定论文片段回答。",
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
    except (APIStatusError, APIConnectionError, RateLimitError) as exc:
        raise LLMServiceError(format_api_error(exc)) from exc
    return response.choices[0].message.content or ""


def build_user_content(prompt: str, image_paths: list[str], max_images: int = 3):
    """构建 OpenAI 兼容的多模态消息；无有效图片时保持纯文本格式。"""
    image_parts: list[dict] = []
    seen: set[str] = set()
    for raw_path in image_paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        normalized = str(path.resolve()) if path.exists() else str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        data_url = image_to_data_url(path)
        if not data_url:
            continue
        image_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url, "detail": "high"},
            }
        )
        if len(image_parts) >= max_images:
            break

    if not image_parts:
        return prompt
    return [
        {"type": "text", "text": prompt},
        *image_parts,
    ]


def image_to_data_url(path: Path, max_bytes: int = 8 * 1024 * 1024) -> str | None:
    if not path.is_file():
        return None
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        return None

    mime_type, _ = mimetypes.guess_type(path.name)
    supported_types = {"image/png", "image/jpeg", "image/webp", "image/gif"}
    if mime_type not in supported_types:
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def is_known_text_only_provider(base_url: str, model: str) -> bool:
    normalized_url = (base_url or "").lower()
    normalized_model = (model or "").lower()
    if "api.deepseek.com" in normalized_url:
        return True
    return normalized_model.startswith(("deepseek-chat", "deepseek-reasoner", "deepseek-v4"))
