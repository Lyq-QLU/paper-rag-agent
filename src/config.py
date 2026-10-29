from dataclasses import dataclass
import json
import os
from pathlib import Path
import tomllib
from filelock import FileLock

from src.session_manager import get_user_dir, atomic_write_text_or_bytes


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


def get_openalex_api_key(user_id: str | None = None) -> str | None:
    return _get_setting("OPENALEX_API_KEY", user_id=user_id)


def save_openalex_api_key(api_key: str, user_id: str) -> None:
    if not api_key.strip():
        raise ValueError("请填写 OpenAlex API Key。")
    _save_settings(get_user_api_config_path(user_id), {"OPENALEX_API_KEY": api_key.strip()})


def _save_settings(path: Path, updates: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        settings = _load_toml(path)
        settings.update(updates)
        atomic_write_text_or_bytes(path, "\n".join(
            f"{key} = {json.dumps(value, ensure_ascii=False)}"
            for key, value in settings.items() if isinstance(value, str)
        ) + "\n")
        os.chmod(path, 0o600)


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
    _save_settings(config_path, {"OPENAI_API_KEY": api_key.strip(),
                                 "OPENAI_BASE_URL": base_url.strip(), "OPENAI_MODEL": model.strip()})


config = get_config()
