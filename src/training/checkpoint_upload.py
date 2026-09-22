"""Bounded asynchronous transfers of atomically replaced checkpoint files."""
from pathlib import Path
import shutil
import tempfile
import threading


class CheckpointUploader:
    """Keep one active upload and the latest pending immutable snapshot.

    Checkpoints must be saved using temporary-file + atomic replacement. Open
    file handles pin those inodes until the worker copies them to local staging.
    Intermediate pending uploads can be superseded; close drains the final one.
    """
    def __init__(self, directory, upload):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.upload = upload
        self.condition = threading.Condition()
        self.pending = None
        self.copy_pending = None
        self.copy_done = False
        self.version = 0
        self.closing = False
        self.error = None
        self.thread = threading.Thread(target=self._worker, daemon=True, name='checkpoint-upload')
        self.copy_thread = threading.Thread(target=self._copy_worker, daemon=True, name='checkpoint-snapshot')
        self.copy_thread.start()
        self.thread.start()

    def check(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError('Background checkpoint upload failed') from self.error

    def submit(self, paths, epoch):
        self.check()
        sources = []
        try:
            for source in paths:
                source = Path(source)
                if source.exists():
                    sources.append((source.name, source.open('rb')))
            with self.condition:
                if self.closing or self.error is not None:
                    raise RuntimeError('Checkpoint uploader is closed') from self.error
                if self.copy_pending is not None:
                    for _, stream in self.copy_pending[0]:
                        stream.close()
                self.version += 1
                self.copy_pending = (sources, epoch, self.version)
                self.condition.notify_all()
        except BaseException:
            for _, stream in sources:
                stream.close()
            raise

    def _fail(self, error):
        with self.condition:
            self.error = error
            self.closing = True
            if self.copy_pending is not None:
                for _, stream in self.copy_pending[0]:
                    stream.close()
                self.copy_pending = None
            if self.pending is not None:
                shutil.rmtree(self.pending[0], ignore_errors=True)
                self.pending = None
            self.condition.notify_all()

    def _copy_worker(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.copy_pending is not None or self.closing)
                if self.copy_pending is None:
                    self.copy_done = True
                    self.condition.notify_all()
                    return
                sources, epoch, version = self.copy_pending
                self.copy_pending = None
            snapshot = None
            try:
                snapshot = Path(tempfile.mkdtemp(prefix=f'epoch-{epoch:03d}-', dir=self.directory))
                paths = []
                for name, stream in sources:
                    target = snapshot / name
                    with target.open('wb') as output:
                        shutil.copyfileobj(stream, output, length=8 * 1024 * 1024)
                    stream.close()
                    paths.append(target)
                with self.condition:
                    if self.error is None and version == self.version:
                        if self.pending is not None:
                            shutil.rmtree(self.pending[0], ignore_errors=True)
                        self.pending = (snapshot, paths, epoch)
                        snapshot = None  # ownership passes to the upload worker
                        self.condition.notify_all()
            except BaseException as error:
                self._fail(error)
            finally:
                for _, stream in sources:
                    stream.close()
                if snapshot is not None:
                    shutil.rmtree(snapshot, ignore_errors=True)

    def _worker(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.copy_done or self.error is not None)
                if self.pending is None:
                    return
                snapshot, paths, epoch = self.pending
                self.pending = None
            try:
                self.upload(paths, epoch)
            except BaseException as error:
                self._fail(error)
                return
            finally:
                shutil.rmtree(snapshot, ignore_errors=True)

    def close(self):
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        self.copy_thread.join()
        self.thread.join()
        self.check()
