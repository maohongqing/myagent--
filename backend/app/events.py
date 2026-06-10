from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import AsyncIterator

from myagent.backend.app.models import AgentRunEvent
from myagent.backend.app.storage import Storage


class EventBus:
    def __init__(self, storage: Storage):
        self.storage = storage
        self._subscribers: dict[str, set[asyncio.Queue[AgentRunEvent]]] = defaultdict(set)

    async def publish(self, event: AgentRunEvent) -> None:
        self.storage.append_event(event)
        for queue in list(self._subscribers[event.task_id]):
            await queue.put(event)

    async def subscribe(self, task_id: str) -> AsyncIterator[AgentRunEvent]:
        queue: asyncio.Queue[AgentRunEvent] = asyncio.Queue()
        self._subscribers[task_id].add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers[task_id].discard(queue)


def format_sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
