"""Upload immutable checkpoint files without retaining CPU training snapshots."""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Thread


@dataclass(frozen=True)
class CheckpointFile:
    path: Path
    step: int
    epoch: int
    size: int
    md5: str
    slots: tuple[str, ...]


class CheckpointUploadQueue:
    """Keep every best winner and only the newest pending last checkpoint.

    The active file is immutable until acknowledged. Superseded last-only files
    are removed from the queue; hard-linked local slots retain their own state.
    """

    def __init__(self, upload, *, retry_delay=None, max_retry_delay=300,
                 retryable=None, on_retry=None):
        self._upload = upload
        self._retry_delay = retry_delay
        self._max_retry_delay = max_retry_delay
        self._retryable = retryable
        self._on_retry = on_retry
        self._condition = Condition()
        self._best = deque()
        self._latest = None
        self._active = None
        self._error = None
        self._closing = False
        self._thread = Thread(target=self._run, name='checkpoint-upload', daemon=True)
        self._thread.start()

    def _check_locked(self):
        if self._error is not None:
            raise RuntimeError('Background checkpoint upload failed; local state is preserved') from self._error

    def check(self):
        with self._condition:
            self._check_locked()

    def submit(self, item):
        discard = None
        with self._condition:
            self._check_locked()
            if self._closing:
                raise RuntimeError('Checkpoint upload queue is closing')
            previous = self._latest
            best_slots = tuple(slot for slot in item.slots if slot != 'last.pt')
            if best_slots:
                self._best.append((item, best_slots))
            if 'last.pt' in item.slots:
                self._latest = item
            if ('last.pt' in item.slots and previous is not None and previous is not self._active
                    and not any(queued is previous for queued, _ in self._best)):
                discard = previous
            self._condition.notify_all()
        if discard is not None:
            discard.path.unlink(missing_ok=True)

    def wait(self):
        with self._condition:
            while self._active is not None or self._latest is not None or self._best:
                self._check_locked()
                self._condition.wait()
            self._check_locked()

    def close(self):
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join()
        self.check()

    def _run(self):
        while True:
            with self._condition:
                while self._latest is None and not self._best and not self._closing:
                    self._condition.wait()
                if self._latest is None and not self._best:
                    return
                if self._best:
                    item, slots = self._best.popleft()
                    if self._latest is item:
                        slots = ('last.pt', *slots)
                        self._latest = None
                else:
                    item, slots = self._latest, ('last.pt',)
                    self._latest = None
                self._active = item
            try:
                attempt = 0
                while True:
                    try:
                        self._upload(item, slots)
                        break
                    except Exception as error:
                        if (self._retry_delay is None
                                or (self._retryable is not None and not self._retryable(error))):
                            raise
                        attempt += 1
                        if self._on_retry is not None:
                            self._on_retry(item, slots, error, attempt)
                        delay = min(self._retry_delay * 2 ** min(attempt - 1, 8),
                                    self._max_retry_delay)
                        with self._condition:
                            self._condition.wait(timeout=delay)
                item.path.unlink(missing_ok=True)
            except BaseException as error:
                with self._condition:
                    self._error = error
                    self._active = None
                    self._condition.notify_all()
                return
            with self._condition:
                self._active = None
                self._condition.notify_all()
