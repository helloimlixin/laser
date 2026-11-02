"""Commit stable W&B checkpoint names and verify their cloud digests."""
import base64
import hashlib
import json
import time
from pathlib import Path


class VerifiedCloudUpload:
    def __init__(self, run_path, receipt):
        self.run_path = run_path
        self.receipt = Path(receipt)
        self.run = None

    def __call__(self, paths, epoch):
        import wandb
        if self.run is None:
            self.run = wandb.Api(timeout=120).run(self.run_path)
        receipts = []
        # Upload metric winners before the larger recovery state.
        for path in sorted(paths, key=lambda p: (p.name == 'last.pt', p.name)):
            with path.open('rb') as stream:
                digest = base64.b64encode(hashlib.file_digest(stream, 'md5').digest()).decode()
            size = path.stat().st_size
            for attempt in range(3):
                try:
                    try:
                        remote = self.run.file(path.name)
                        unchanged = remote.size == size and remote.md5 == digest
                    except (ValueError, IndexError):
                        unchanged = False
                    if not unchanged:
                        self.run.upload_file(str(path), root=str(path.parent))
                    for check in range(12):
                        remote = self.run.file(path.name)
                        if remote.size == size and remote.md5 == digest:
                            break
                        if check == 11:
                            raise RuntimeError(f'Cloud checkpoint verification failed: {path.name}')
                        time.sleep(5)
                    receipts.append(dict(name=path.name, bytes=size, md5=digest,
                                         updated_at=remote.updated_at, uploaded=not unchanged))
                    record = dict(epoch=epoch, verified_unix=time.time(), files=receipts,
                                  complete=len(receipts) == len(paths))
                    self.receipt.parent.mkdir(parents=True, exist_ok=True)
                    temporary = self.receipt.with_suffix('.tmp')
                    temporary.write_text(json.dumps(record, indent=2) + '\n')
                    temporary.replace(self.receipt)
                    print('Verified W&B checkpoint file: ' + path.name, flush=True)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(5 * (attempt + 1))
        record = dict(epoch=epoch, verified_unix=time.time(), files=receipts, complete=True)
        self.receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(self.receipt)
        print('Verified W&B checkpoint files: ' + ', '.join(row['name'] for row in receipts), flush=True)
