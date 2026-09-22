"""Report worker failures without entering collective teardown alone."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback


def fatal_worker_error(error, output, rank):
    record = dict(type=type(error).__name__, message=str(error), time=time.time(),
                  rank=rank, pid=os.getpid(), traceback=traceback.format_exc())
    text = json.dumps(record, indent=2) + '\n'
    # The output filesystem may itself have caused the exception. Always leave
    # an independent local diagnostic before attempting the normal receipt.
    emergency = Path(tempfile.gettempdir())/'laser-worker-failures'/f'{os.getpid()}-rank{rank}.json'
    for path in (emergency, Path(output)/f'failure-rank{rank}.json'):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(text)
            temporary.replace(path)
        except OSError:
            pass
    try:
        sys.stderr.write(text)
        sys.stderr.flush()
    except OSError:
        pass
    # Other ranks may be inside NCCL collectives. Teardown or W&B finish here
    # can hang before torchrun learns that this rank failed. Let torchrun stop
    # peers and let the supervisor resume the last complete checkpoint.
    os._exit(1)
