from unittest.mock import patch


def test_health(client) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_returns_sse_from_the_compiled_graph(client, device_headers) -> None:
    response = client.post(
        "/api/v1/chat",
        headers=device_headers,
        json={"thread_id": "sse-thread", "message": "她说今天很累"},
    )
    assert response.status_code == 200
    assert "event: start" in response.text
    assert "event: complete" in response.text
    assert '"skill": "goutoujunshi"' in response.text
    assert "event: token" not in response.text


def test_invalid_device_id_is_rejected(client) -> None:
    response = client.get(
        "/api/v1/threads",
        headers={"X-Device-Id": "not-a-uuid"},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "无效设备标识"}


def test_device_private_thread_lifecycle(client, device_headers) -> None:
    thread_id = "private-thread"
    stranger = {"X-Device-Id": "467c1c91-65b3-4f47-a9f7-f45c4321ac87"}
    client.post(
        "/api/v1/chat",
        headers=device_headers,
        json={"thread_id": thread_id, "message": "怎么回"},
    )
    assert client.get(
        f"/api/v1/threads/{thread_id}",
        headers=stranger,
    ).status_code == 404
    assert client.post(
        "/api/v1/chat",
        headers=stranger,
        json={"thread_id": thread_id, "message": "继续聊"},
    ).status_code == 404
    assert client.delete(
        f"/api/v1/threads/{thread_id}",
        headers=device_headers,
    ).json() == {"deleted": True}


def test_sse_error_hides_internal_detail(client, device_headers) -> None:
    graph = client.app.state.chat_service._graph
    with patch.object(
        graph,
        "invoke",
        side_effect=RuntimeError("internal endpoint detail"),
    ):
        response = client.post(
            "/api/v1/chat",
            headers=device_headers,
            json={"thread_id": "error-thread", "message": "怎么回"},
        )
    assert "服务暂时不可用" in response.text
    assert "internal endpoint detail" not in response.text
