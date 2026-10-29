import httpx
from openai import APITimeoutError

from src.llm import format_api_error


def test_model_timeout_has_distinct_message():
    error = APITimeoutError(request=httpx.Request("POST", "https://example.com/chat/completions"))
    assert "响应超时" in format_api_error(error)
    assert "Base URL" not in format_api_error(error)
