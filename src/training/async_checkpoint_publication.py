"""Publish committed checkpoint files without blocking GPU training."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor


class AsyncCheckpointPublisher:
    """Bound publication work and propagate commit/upload failures to callers."""

    def __init__(self, max_pending=2):
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint-publish")
        self._pending = deque()
        self._max_pending = max_pending
        self._closed = False

    def check(self):
        while self._pending and self._pending[0].done():
            self._pending.popleft().result()

    def submit(self, dependency, function):
        self.check()
        if self._closed:
            raise RuntimeError("checkpoint publisher is closed")
        if len(self._pending) >= self._max_pending:
            self._pending.popleft().result()

        def publish():
            if dependency is not None:
                dependency.result()
            return function()

        self._pending.append(self._executor.submit(publish))

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            while self._pending:
                self._pending.popleft().result()
        finally:
            self._executor.shutdown(wait=True)
