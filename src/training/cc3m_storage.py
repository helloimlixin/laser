"""Local run artifacts with retryable, asynchronous workspace publication."""
import os
from pathlib import Path
import shutil
import sys
import threading


def local_run_root(cfg):
    return Path(cfg['local_checkpoints']).parent/'live'


class WorkspaceMirror:
    """Only this daemon touches the workspace; training never waits for it."""

    def __init__(self, mappings, interval=30):
        self.mappings = [(Path(source), Path(target)) for source, target in mappings]
        self.interval = interval
        self.published = {}
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name='workspace-mirror', daemon=True)

    def start(self):
        self.thread.start()
        return self

    @staticmethod
    def _directory(path):
        # Earlier launchers published local directories as workspace symlinks.
        # Replace those aliases so publication cannot write back into local data.
        if path.is_symlink():
            path.unlink()
        elif path.is_file():
            path.rename(path.with_name(path.name+'.before-local-mirror'))
        path.mkdir(parents=True, exist_ok=True)

    def sync_once(self):
        failures = []
        for source_root, target_root in self.mappings:
            for source in sorted(source_root.rglob('*'), key=lambda p: (p.suffix == '.json', str(p))):
                target = target_root/source.relative_to(source_root)
                try:
                    if not source.is_file() or source.name.endswith(('.tmp', '.partial')):
                        continue
                    stat = source.stat()
                    identity = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
                    if self.published.get(source) == identity:
                        continue
                    self._directory(target_root)
                    for parent in reversed(target.relative_to(target_root).parents):
                        if parent != Path('.'):
                            self._directory(target_root/parent)
                    temporary = target.with_name(target.name+f'.mirror-{os.getpid()}.tmp')
                    shutil.copyfile(source, temporary)
                    latest = source.stat()
                    if (temporary.stat().st_size != stat.st_size or
                            (latest.st_ino, latest.st_size, latest.st_mtime_ns) != identity):
                        # Logs may grow during copying; publish them on the next pass.
                        temporary.unlink(missing_ok=True)
                        continue
                    temporary.replace(target)
                    self.published[source] = identity
                except OSError as error:
                    failures.append(f'{target}: {error}')
                    # In particular, never publish a checkpoint receipt after
                    # its payload failed to copy. Retry the mapping next pass.
                    break
        return failures

    def _run(self):
        while True:
            failures = self.sync_once()
            if failures:
                print(f'Workspace publication deferred; local artifacts retained: {failures[0]}',
                      file=sys.stderr, flush=True)
            if self.stopping.wait(self.interval):
                self.sync_once()
                return

    def close(self):
        self.stopping.set()
        # A stalled filesystem must never hold up training or shutdown.
        self.thread.join(timeout=1)
