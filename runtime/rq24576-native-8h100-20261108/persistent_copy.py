"""Copy atomic checkpoint payloads concurrently across independent file ranges."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path


def copy_file(source, destination, workers=16):
    source, destination = Path(source), Path(destination)
    source_fd = os.open(source, os.O_RDONLY)
    target_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    size = os.fstat(source_fd).st_size
    block = 32 * 1024**2
    os.ftruncate(target_fd, size)

    def segment(first):
        stop = min(first + block, size)
        while first < stop:
            data = os.pread(source_fd, min(4 * 1024**2, stop - first), first)
            if not data:
                raise OSError('Incomplete source checkpoint read')
            view = memoryview(data)
            while view:
                count = os.pwrite(target_fd, view, first)
                if count <= 0:
                    raise OSError('Incomplete persistent checkpoint write')
                first += count
                view = view[count:]

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(segment, range(0, size, block)))
    finally:
        os.close(source_fd)
        os.close(target_fd)
    assert destination.stat().st_size == size
