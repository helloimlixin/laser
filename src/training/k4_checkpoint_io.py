"""Validated immutable checkpoint persistence from the September 27 runtime."""
import gc, hashlib, json, os, shutil, tempfile, time
from pathlib import Path
import torch

def _checkpoint_file_identity(path: Path):
    stat = path.stat()
    return {key: getattr(stat, key) for key in
            ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}

def _local_checkpoint_paths(source: Path):
    root = os.environ.get("LASER_CHECKPOINT_UPLOAD_CACHE_DIR")
    if not root:
        return None
    key = hashlib.sha256(str(source.resolve()).encode()).hexdigest()
    local = Path(root).expanduser() / "objects" / (key + ".pt")
    return local, local.with_suffix(".json")

def _checkpoint_upload_source(source: Path):
    paths = _local_checkpoint_paths(source)
    if paths is not None:
        local, receipt = paths
        try:
            metadata = json.loads(receipt.read_text())
            identity = _checkpoint_file_identity(source)
            matches = metadata["identity"] == identity
            resolved = source.resolve()
            if (not matches
                    and os.environ.get("LASER_CHECKPOINT_IMMUTABLE_FILES") == "1"
                    and resolved.parent.name == ".checkpoint-data"
                    and metadata.get("source") == str(resolved)):
                # Object-backed storage may finalize ctime after commit. The
                # unique immutable payload was already compared with the local
                # serialization before its receipt was written. A ctime-only
                # change does not require rereading this multi-GB payload.
                stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
                matches = all(metadata["identity"].get(key) == identity[key]
                              for key in stable)
            if matches and local.stat().st_size == identity["st_size"]:
                return local
        except (OSError, ValueError, KeyError):
            pass
    return source

def atomic_torch_save(payload, target: Path, *, background=None, on_commit=None):
    """Persist a checkpoint without exposing a partial destination file.

    Network filesystems can fail in the middle of PyTorch's multi-gigabyte zip
    serialization.  When ``LASER_CHECKPOINT_STAGING_DIR`` is set, serialize
    once to local storage and retry only the copy to the persistent filesystem.
    The existing checkpoint remains untouched until a complete copy is ready.
    Optional background persistence starts only after serialization completes,
    so training cannot mutate tensors being written by the worker.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    staging_root = os.environ.get("LASER_CHECKPOINT_STAGING_DIR")
    if background is not None:
        if not staging_root:
            raise ValueError("background checkpoint copies require local staging")
        background.wait()
    if staging_root:
        local_dir = Path(staging_root).expanduser()
        local_dir.mkdir(parents=True, exist_ok=True)
        file_descriptor, local_name = tempfile.mkstemp(
            prefix=f"{target.name}.", suffix=".staging", dir=local_dir
        )
        os.close(file_descriptor)
        serialized = Path(local_name)
    else:
        serialized = temporary

    try:
        torch.save(payload, serialized)
    except BaseException:
        serialized.unlink(missing_ok=True)
        raise
    finally:
        # PyTorch pickler cycles can retain the saved CUDA storage table.
        gc.collect()

    def commit():
        _persist_serialized_checkpoint(serialized, target)
        if on_commit is not None:
            on_commit()

    if background is None:
        commit()
    else:
        try:
            background.submit(commit)
        except BaseException:
            serialized.unlink(missing_ok=True)
            raise

def _persist_serialized_checkpoint(serialized: Path, target: Path):
    """Commit an immutable file, retaining the previous checkpoint on failure."""
    if os.environ.get("LASER_CHECKPOINT_IMMUTABLE_FILES") == "1":
        return _persist_immutable_checkpoint(serialized, target)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        if serialized == temporary:
            os.replace(temporary, target)
            return

        serialized_size = serialized.stat().st_size
        copy_attempts = 3
        for attempt in range(1, copy_attempts + 1):
            try:
                temporary.unlink(missing_ok=True)
                shutil.copyfile(serialized, temporary)
                copied_size = temporary.stat().st_size
                if copied_size != serialized_size:
                    raise OSError(
                        f"checkpoint copy size mismatch: local={serialized_size}, "
                        f"persistent={copied_size}"
                    )
                os.replace(temporary, target)
                # Keep the already serialized local inode for W&B. Uploading
                # from shared storage otherwise rereads and copies 17 GB.
                paths = _local_checkpoint_paths(target)
                if paths is not None:
                    local, receipt = paths
                    _replace_hard_link(serialized, local)
                    metadata = {"source": str(target.resolve()),
                                "identity": _checkpoint_file_identity(target)}
                    receipt_tmp = receipt.with_suffix(".json.tmp")
                    receipt_tmp.write_text(json.dumps(metadata) + "\n")
                    os.replace(receipt_tmp, receipt)
                break
            except OSError as error:
                temporary.unlink(missing_ok=True)
                if attempt == copy_attempts:
                    raise
                delay = 5 * attempt
                print(
                    f"Checkpoint copy attempt {attempt}/{copy_attempts} failed: "
                    f"{error}; retrying in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
    finally:
        if serialized != temporary:
            serialized.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)

def _persist_immutable_checkpoint(serialized: Path, target: Path):
    """Avoid replacing large payloads on object-backed shared filesystems.

    Payload names are immutable. Only a small symlink is atomically replaced;
    an interrupted copy leaves the previous checkpoint readable.
    """
    folder = target.parent / '.checkpoint-data'
    folder.mkdir(parents=True, exist_ok=True)
    previous = target.resolve() if target.is_symlink() else None
    payload = folder / f'{target.stem}-{os.getpid()}-{time.time_ns()}.pt'
    link = target.with_suffix(target.suffix + '.link.tmp')
    committed = False
    try:
        shutil.copyfile(serialized, payload)
        if payload.stat().st_size != serialized.stat().st_size:
            raise OSError('Immutable checkpoint copy size mismatch')
        # Probe both ends after close: this mount can stat an unreadable payload.
        with serialized.open('rb') as local, payload.open('rb') as persistent:
            if local.read(64) != persistent.read(64):
                raise OSError('Immutable checkpoint header mismatch')
            local.seek(-64, 2)
            persistent.seek(-64, 2)
            if local.read(64) != persistent.read(64):
                raise OSError('Immutable checkpoint footer mismatch')
        link.unlink(missing_ok=True)
        link.symlink_to(payload.relative_to(target.parent))
        os.replace(link, target)
        committed = True
        paths = _local_checkpoint_paths(target)
        if paths is not None:
            local, receipt = paths
            _replace_hard_link(serialized, local)
            receipt.write_text(json.dumps({'source': str(target.resolve()),
                'identity': _checkpoint_file_identity(target)}) + '\n')
        if previous is not None and previous.parent == folder and previous != payload:
            old_paths = _local_checkpoint_paths(previous)
            previous.unlink(missing_ok=True)
            for path in old_paths or ():
                path.unlink(missing_ok=True)
    finally:
        link.unlink(missing_ok=True)
        if not committed:
            payload.unlink(missing_ok=True)
        serialized.unlink(missing_ok=True)

def remove_checkpoint(path: Path):
    """Remove an owned immutable payload together with its named checkpoint."""
    payload = path.resolve() if path.is_symlink() else None
    path.unlink(missing_ok=True)
    if payload is not None and payload.parent == path.parent / '.checkpoint-data':
        paths = _local_checkpoint_paths(payload)
        payload.unlink(missing_ok=True)
        for local in paths or ():
            local.unlink(missing_ok=True)

def _replace_hard_link(source: Path, destination: Path):
    """Atomically point a fixed upload slot at an immutable checkpoint inode."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

def upload_selected_checkpoint_files(
    wb,
    *,
    last_checkpoint: Path,
    best_fid,
    upload_dir: Path,
    best_inception=(),
):
    """Replace fixed W&B slots with last and the best FID/IS checkpoints.

    Artifact versions are immutable and therefore accumulate indefinitely.
    Run files keep stable online names, matching the Stage-1 retention policy,
    while the local hard links avoid another multi-gigabyte checkpoint copy.
    """
    sources = [("last.pt", last_checkpoint)]
    sources.extend(
        (f"best-fid-{rank_index:02d}.pt", Path(saved_path))
        for rank_index, (_, saved_path) in enumerate(
            sorted(best_fid, key=lambda item: float(item[0]))[:3], start=1
        )
    )
    sources.extend(
        (f"best-is-{rank_index:02d}.pt", Path(saved_path))
        for rank_index, (_, saved_path) in enumerate(
            sorted(best_inception, key=lambda item: float(item[0]), reverse=True)[:3], start=1
        )
    )
    sources = [(slot, source.resolve()) for slot, source in sources if source.is_file()]
    if not sources:
        return []

    upload_dir = upload_dir.expanduser().resolve()
    local_root = os.environ.get("LASER_CHECKPOINT_UPLOAD_CACHE_DIR")
    if local_root:
        key = hashlib.sha256(str(upload_dir).encode()).hexdigest()
        upload_dir = Path(local_root).expanduser() / "slots" / key
    upload_dir.mkdir(parents=True, exist_ok=True)
    active_names = {slot for slot, _ in sources}
    for pattern in ("best-fid-*.pt", "best-is-*.pt"):
        for stale in upload_dir.glob(pattern):
            if stale.name not in active_names:
                stale.unlink(missing_ok=True)

    uploaded = []
    for slot, source in sources:
        destination = upload_dir / slot
        _replace_hard_link(_checkpoint_upload_source(source), destination)
        if os.environ.get("LASER_CHECKPOINT_DIRECT_UPLOAD") != "1":
            wb.save(str(destination), base_path=str(upload_dir), policy="now")
        uploaded.append(destination)
    print(
        "Queued fixed W&B checkpoint files: "
        + ", ".join(path.name for path in uploaded),
        flush=True,
    )
    return uploaded
