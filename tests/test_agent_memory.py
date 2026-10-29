from src.agent_memory import AgentMemoryStore


def test_profile_is_user_isolated_and_can_be_cleared(tmp_path):
    store = AgentMemoryStore(tmp_path / "memory.sqlite3")
    store.remember("alice", {"research_focus": "routing"})
    assert store.get_profile("alice")["preferences"]["research_focus"] == "routing"
    assert store.get_profile("bob")["preferences"] == {}
    store.clear_profile("alice")
    assert store.get_profile("alice")["preferences"] == {}
    store.close()


def test_pending_action_is_idempotently_resolved(tmp_path):
    store = AgentMemoryStore(tmp_path / "memory.sqlite3")
    store.create_pending(
        action_id="a1", user_id="u1", thread_id="t1",
        action_type="ingest", payload={"papers": [1, 2]},
    )
    first = store.resolve_pending("a1", {"selected": [1]})
    second = store.resolve_pending("a1", {"selected": [2]})
    assert first["status"] == "resolved"
    assert second["payload"]["decision"]["selected"] == [1]
    store.close()
