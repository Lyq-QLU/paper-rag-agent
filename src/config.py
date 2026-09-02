from dataclasses import dataclass
import json
import os
from pathlib import Path
import tomllib

from src.session_manager import get_user_dir


SECRETS_PATH = Path(".streamlit/secrets.toml")


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}

    with path.open("rb") as file:
        return tomllib.load(file)


def _load_streamlit_secrets() -> dict:
    return _load_toml(SECRETS_PATH)


def get_user_api_config_path(user_id: str) -> Path:
    return get_user_dir(user_id) / "api_config.toml"


def _get_setting(name: str, default: str | None = None, user_id: str | None = None) -> str | None:
    env_value = os.getenv(name)
    if env_value:
        return env_value

    if user_id:
        user_secrets = _load_toml(get_user_api_config_path(user_id))
        user_value = user_secrets.get(name)
        if user_value:
            return str(user_value)

    secrets = _load_streamlit_secrets()
    value = secrets.get(name)
    if value:
        return str(value)

    return default


@dataclass(frozen=True)
class RAGConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 200
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    retrieval_mode: str = "hybrid"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"


def get_config(user_id: str | None = None) -> RAGConfig:
    return RAGConfig(
        retrieval_mode=_get_setting("RETRIEVAL_MODE", "hybrid", user_id=user_id) or "hybrid",
        openai_api_key=_get_setting("OPENAI_API_KEY", user_id=user_id),
        openai_base_url=_get_setting("OPENAI_BASE_URL", "https://api.openai.com/v1", user_id=user_id) or "https://api.openai.com/v1",
        openai_model=_get_setting("OPENAI_MODEL", "gpt-4o-mini", user_id=user_id) or "gpt-4o-mini",
    )


def save_api_config(api_key: str, base_url: str, model: str, user_id: str | None = None) -> None:
    config_path = get_user_api_config_path(user_id) if user_id else SECRETS_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"OPENAI_API_KEY = {json.dumps(api_key.strip(), ensure_ascii=False)}",
        f"OPENAI_BASE_URL = {json.dumps(base_url.strip(), ensure_ascii=False)}",
        f"OPENAI_MODEL = {json.dumps(model.strip(), ensure_ascii=False)}",
        "",
    ]
    config_path.write_text("\n".join(lines), encoding="utf-8")


config = get_config()
