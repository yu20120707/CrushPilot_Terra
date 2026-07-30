import asyncio
import unittest

from app.api.v1.routes import ClosingStreamingResponse


class _Events:
    def __init__(self):
        self.closed = False

    def __iter__(self):
        return iter(["event: start\ndata: {}\n\n"])

    def close(self):
        self.closed = True


class SseCleanupTests(unittest.TestCase):
    def test_disconnect_before_stream_iteration_closes_events(self):
        events = _Events()
        response = ClosingStreamingResponse(events, media_type="text/event-stream")

        async def receive():
            return {"type": "http.disconnect"}

        async def send(_):
            return None

        asyncio.run(
            response(
                {"type": "http", "asgi": {"version": "3.0"}, "method": "GET", "path": "/"},
                receive,
                send,
            )
        )
        self.assertTrue(events.closed)
