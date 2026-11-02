"""Bounded checkpoint I/O after the caller has serialized an immutable file."""

from concurrent.futures import ThreadPoolExecutor


class BackgroundCheckpointWriter:
    """Keep at most one pending write and surface failures on the caller."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint-copy")
        self._pending = None

    def check(self):
        if self._pending is not None and self._pending.done():
            self._pending.result()

    @property
    def completion(self):
        """Capture the current commit without blocking the training thread."""
        return self._pending

    def append(self, function):
        """Queue file-only work behind the current immutable snapshot.

        Full tensor snapshots still use ``submit`` to bound their memory and
        disk usage. Dependent aliases can be published in order without
        making the caller wait for shared storage.
        """
        previous = self._pending

        def ordered():
            if previous is not None:
                previous.result()
            return function()

        self._pending = self._executor.submit(ordered)

    def wait(self):
        if self._pending is not None:
            pending, self._pending = self._pending, None
            pending.result()

    def submit(self, function):
        self.wait()
        self._pending = self._executor.submit(function)

    def close(self):
        try:
            self.wait()
        finally:
            self._executor.shutdown(wait=True)
