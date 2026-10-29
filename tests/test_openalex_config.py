from src import config, session_manager
from src.paper_library import source_error_message


def test_openalex_key_is_saved_per_user_and_preserves_model_config(tmp_path, monkeypatch):
    monkeypatch.setattr(session_manager, "DATA_ROOT", tmp_path / "users")
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    config.save_api_config("synthetic-qwen-key", "https://example.com/v1", "test-model", user_id="alice")
    config.save_openalex_api_key("synthetic-openalex-key", "alice")
    assert config.get_openalex_api_key("alice") == "synthetic-openalex-key"
    assert config.get_openalex_api_key("bob") is None
    assert config.get_config("alice").openai_model == "test-model"
    config.save_api_config("new-synthetic-key", "https://example.com/v1", "other-model", user_id="alice")
    assert config.get_openalex_api_key("alice") == "synthetic-openalex-key"


def test_openalex_limit_messages_are_actionable():
    assert "论文来源设置" in source_error_message("openalex", "HTTP 429")
    assert "额度重置" in source_error_message("openalex", "not_enough_credits; API Key")
