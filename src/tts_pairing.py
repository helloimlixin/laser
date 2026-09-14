"""Verifiable common initialization and data order for the audio prior control."""
import hashlib
import json
from pathlib import Path

import torch


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def common_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if not k.startswith(('fields.', 'heads.'))}


def state_sha(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def record_signature(cache,include_frames=True):
    """Hash utterance identity/text/split; optionally include codec frame counts."""
    records = [{**{k: v for k, v in r.items() if k != 'codes'}, **({'frames':len(r['codes'])} if include_frames else {})}
               for r in cache['records']]
    identity = {key: cache[key] for key in ('phone_to_id', 'speaker_to_id')}
    identity['records'] = records
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def batch_chain(previous, batch):
    digest = hashlib.sha256(bytes.fromhex(previous))
    for key in ('indices', 'phones', 'lengths', 'text_lengths', 'speakers'):
        value = (batch.get('audit_lengths',batch[key]) if key=='lengths' else batch[key]).detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_paired_initialization(model, cache, cache_path, config):
    protocol = json.loads(Path(config['protocol']).read_text())
    if file_sha(config['protocol']) != config['protocol_sha256']:
        raise ValueError('Paired protocol changed')
    arm = model.cfg.codec
    expected = protocol['arms'][arm]
    if file_sha(cache_path) != expected['cache_sha256'] or record_signature(cache,include_frames=protocol.get('record_signature_basis')!='utterance_metadata') != protocol['record_signature']:
        raise ValueError('Paired cache content or utterance alignment changed')
    if cache['codec_sha256'] != expected['codec_sha256']:
        raise ValueError('Frozen codec changed')
    if file_sha(expected['initialization']) != expected['initialization_sha256']:
        raise ValueError('Paired initialization changed')
    state = torch.load(expected['initialization'], map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    if state_sha(common_state(model)) != protocol['common_initialization_sha256']:
        raise ValueError('Shared text, temporal, or depth weights differ')
    return protocol
