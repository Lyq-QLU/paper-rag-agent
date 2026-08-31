from fastapi.testclient import TestClient

from api import app
import src.session_manager as session_manager


def test_health_and_profile_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(session_manager, "DATA_ROOT", tmp_path / "users")
    # api imported helpers still resolve DATA_ROOT through the module globals.
    client = TestClient(app)
    assert client.get("/api/health").json() == {"status": "ok"}
    profile = client.get("/api/user-profile", params={"user_id": "api-user"}).json()
    assert profile["enabled"] is True
    updated = client.patch(
        "/api/user-profile/settings", params={"user_id": "api-user"},
        json={"enabled": False, "ttl_days": 30},
    ).json()
    assert updated["enabled"] is False
    assert updated["ttl_days"] == 30
