#!/usr/bin/env python3
"""Prepare immutable shared-physical-grid caches; never migrate or train a model."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import torch

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from src import shared_physical_coefficients as codec


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def checked_copy(source, destination, expected_sha=None):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = expected_sha or sha256(source)
    if destination.exists():
        if sha256(destination) != digest:
            raise ValueError(f'refusing to replace a different durable file: {destination}')
    else:
        temporary = destination.with_name(destination.name + '.copy-tmp')
        shutil.copyfile(source, temporary)
        if sha256(temporary) != digest:
            raise ValueError(f'durable copy verification failed: {destination}')
        temporary.replace(destination)
    return dict(path=str(destination), sha256=digest, bytes=destination.stat().st_size)


def verify_conversion(source, converted, centers, chunk_images=1024):
    """Check every support, label and physical coefficient, with bounded scratch."""
    if not torch.equal(source['labels'], converted['labels']):
        raise ValueError('conversion changed image labels')
    scales = torch.tensor(source['meta']['coeff_scales'], dtype=torch.float32)
    minima = torch.full_like(scales, torch.inf)
    maxima = torch.full_like(scales, -torch.inf)
    for start in range(0, len(source['atoms']), chunk_images):
        stop = start + chunk_images
        if not torch.equal(source['atoms'][start:stop], converted['atoms'][start:stop]):
            raise ValueError('conversion changed sparse support or order')
        expected = source['coeffs'][start:stop] * scales
        actual = converted['coeffs'][start:stop]
        if actual.dtype != torch.float32 or not torch.equal(expected, actual):
            raise ValueError('physical coefficients differ from exactly one FP32 conversion')
        if not bool(torch.isfinite(actual).all()):
            raise ValueError('nonfinite physical coefficient')
        flat = actual.reshape(-1, actual.shape[-1])
        minima = torch.minimum(minima, flat.min(0).values)
        maxima = torch.maximum(maxima, flat.max(0).values)
    if bool((minima < centers[0]).any()) or bool((maxima > centers[-1]).any()):
        raise ValueError('shared grid does not cover all physical coefficients')
    codec.require_shared_physical_cache(converted['meta'], centers)
    return dict(all_supports_and_order_exact=True, all_labels_exact=True,
        all_physical_coefficients_exact_fp32_once=True, all_finite=True,
        clipping=False, values_outside_shared_grid=0,
        physical_min_by_depth=minima.tolist(), physical_max_by_depth=maxima.tolist())


def check_cpu_aux(helper_path, tokenizer, tokenizer_sha256, payload, centers):
    """Exercise actual historical pair embedding code with explicitly new buffers."""
    spec = importlib.util.spec_from_file_location('shared_physical_cpu_recovery', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    aux = helper.build_aux(tokenizer, [1.] * payload['atoms'].shape[-1], decoder=False, device='cpu')
    # This is a new auxiliary object. The frozen historical helper itself is untouched.
    aux.coeff_bins = centers.clone()
    aux.coeff_max, aux.coeff_scale = float(centers[-1]), 1.
    aux.soft_target_physical, aux.clamp_coeffs = True, False
    codec.require_shared_physical_cache(payload['meta'], aux.coeff_bins)
    indices = torch.linspace(0, len(payload['atoms']) - 1, min(8, len(payload['atoms']))).long()
    atoms, coefficients = payload['atoms'][indices], payload['coeffs'][indices]
    ids = codec.nearest_coefficient_ids(coefficients, centers)
    expected = aux.dictionary.T[atoms.long()] * centers[ids][..., None]
    actual = aux.compound_embeddings(atoms, ids)
    continuous = aux.physical_contributions(atoms, coefficients)
    expected_continuous = aux.dictionary.T[atoms.long()] * coefficients[..., None]
    if not torch.equal(expected, actual) or not torch.equal(expected_continuous, continuous):
        raise ValueError('actual recovered auxiliary applies incorrect physical units')
    return dict(passed=True, device='cpu', images=len(indices),
        helper_path=str(helper_path), helper_sha256=sha256(helper_path),
        tokenizer_path=str(tokenizer), tokenizer_sha256=tokenizer_sha256,
        all_variants_preserved=atoms.ndim == 5,
        quantized_pair_vectors_exact=True, continuous_pair_vectors_exact=True,
        decoder_invoked=False, model_or_optimizer_created=False,
        note='Fresh auxiliary object with explicit common centers and unit scales; historical source unchanged')


def prepare(args):
    started = time.time()
    torch.set_num_threads(args.cpu_threads)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.durable and args.durable.resolve() == output:
        raise ValueError('local output and durable output must be different directories')
    audit = json.loads(args.source_audit.read_text())
    ranges = json.loads(args.range_audit.read_text())
    if audit.get('passed') is not True:
        raise ValueError('a passed independent source/range audit is required')
    audited_sources = {}
    for path, record in audit['sources'].items():
        identity = str(Path(path).resolve())
        if identity in audited_sources and audited_sources[identity] != record:
            raise ValueError('audit contains inconsistent identities for one resolved source')
        audited_sources[identity] = record
    bound = float(ranges['bound'])
    if bound != float(audit['recommended_grid']['upper']):
        raise ValueError('independent range audits disagree')
    centers = torch.linspace(-bound, bound, 2048, dtype=torch.float32)
    initial_sources = {str(Path(__file__).resolve()): sha256(__file__),
                       str(Path(codec.__file__).resolve()): sha256(codec.__file__)}
    tokenizer_sha256 = None
    if args.recovery_helper:
        tokenizer_record = audited_sources.get(str(args.tokenizer.resolve()))
        tokenizer_sha256 = sha256(args.tokenizer)
        if (not tokenizer_record or tokenizer_record['sha256'] != tokenizer_sha256
                or tokenizer_record['bytes'] != args.tokenizer.stat().st_size):
            raise ValueError('tokenizer differs from the independently audited frozen stage one')
    inputs = {'canonical': args.canonical.resolve(), 'bank': args.bank.resolve()}
    files, records, common_meta = [], {}, None
    for name, path in inputs.items():
        destination = output / f'{name}-shared-physical.pt'
        if destination.exists():
            raise FileExistsError(f'refusing to replace existing prepared cache: {destination}')
        expected = audited_sources.get(str(path))
        if not expected or sha256(path) != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise ValueError(f'source differs from independently audited cache: {path}')
        print(f'Converting verified {name} cache on CPU', flush=True)
        source = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        if tokenizer_sha256 and source['meta'].get('stage1_checkpoint_sha256') != tokenizer_sha256:
            raise ValueError('source cache and verified tokenizer identities differ')
        converted = codec.convert_cache_payload(source, source_sha256=expected['sha256'],
                                                centers=centers, additional_bound=bound)
        validation = verify_conversion(source, converted, centers)
        if common_meta is None:
            common_meta = converted['meta']
        elif common_meta['coefficient_codec_identity'] != converted['meta']['coefficient_codec_identity']:
            raise ValueError('canonical and bank coefficient codec identities differ')
        cpu_aux = None
        if args.recovery_helper:
            cpu_aux = check_cpu_aux(args.recovery_helper, args.tokenizer, tokenizer_sha256,
                                    converted, centers)
        temporary = destination.with_name(destination.name + '.tmp')
        torch.save(converted, temporary)
        temporary.replace(destination)
        # Verify serialized tensors too, rather than only the in-memory conversion.
        saved = torch.load(destination, map_location='cpu', mmap=True, weights_only=False)
        verify_conversion(source, saved, centers)
        record = dict(source=str(path), source_sha256=expected['sha256'],
            output=str(destination), output_sha256=sha256(destination),
            output_bytes=destination.stat().st_size, shape=list(saved['atoms'].shape),
            coefficient_storage=str(saved['coeffs'].dtype),
            coefficient_codec_identity=saved['meta']['coefficient_codec_identity'],
            cache_identity=saved['meta']['bank_identity'],
            invalidated_payload_keys=saved['meta']['invalidated_coefficient_payload_keys'],
            validation=validation, recovered_cpu_aux=cpu_aux)
        records[name] = record
        files.append(destination)
        print(f'Prepared and verified {name}: {record["output_bytes"]} bytes', flush=True)
        del source, converted, saved
        gc.collect()
    legacy_rejected = None
    if args.legacy_checkpoint:
        checkpoint = torch.load(args.legacy_checkpoint, map_location='cpu', mmap=True, weights_only=False)
        try:
            codec.require_compatible_checkpoint_codec(checkpoint['config'], common_meta)
        except ValueError as error:
            legacy_rejected = dict(passed=True, checkpoint=str(args.legacy_checkpoint),
                checkpoint_sha256=sha256(args.legacy_checkpoint),
                step=checkpoint.get('global_step'), reason=str(error),
                weights_loaded_into_model=False, checkpoint_mutated=False)
        else:
            raise ValueError('legacy checkpoint was incorrectly accepted without vocabulary migration')
        del checkpoint
    overrides = {key: common_meta[key] for key in (
        'coefficient_representation', 'coefficient_units', 'coefficient_normalization',
        'coefficient_storage', 'coefficient_quantizer', 'coefficient_codec_identity',
        'coeff_bin_centers', 'coeff_bin_centers_sha256', 'coeff_scales', 'coeff_scale',
        'coeff_max', 'coeff_vocab_size', 'clip_coefficients')}
    overrides.update(coeff_target_space='physical', soft_target_physical=True,
        clamp_coeffs=False, requires_stage2_vocabulary_migration=True,
        legacy_stage2_checkpoint_compatible=False)
    overrides_path = output / 'codec-overrides.json'
    write_json(overrides_path, overrides)
    files.append(overrides_path)
    source_dir = output / 'prepared-source'
    source_dir.mkdir(exist_ok=True)
    for path, digest in initial_sources.items():
        if sha256(path) != digest:
            raise ValueError(f'conversion source changed during execution: {path}')
        frozen = source_dir / Path(path).name
        checked_copy(path, frozen, digest)
        files.append(frozen)
    for path in (args.source_audit.resolve(), args.range_audit.resolve()):
        frozen = output / path.name
        if path != frozen:
            checked_copy(path, frozen)
        files.append(frozen)
    report = dict(passed=True, created_unix=time.time(), elapsed_seconds=time.time()-started,
        scope='CPU data/grid preparation only; no model migration, training, sampling or publication',
        grid=dict(lower=-bound, upper=bound, count=2048,
                  ideal_spacing=2*bound/2047, scales=common_meta['coeff_scales'],
                  coefficient_codec_identity=common_meta['coefficient_codec_identity'],
                  centers_sha256=common_meta['coeff_bin_centers_sha256']),
        caches=records, source_files_sha256=initial_sources,
        source_audit_sha256=sha256(args.source_audit), range_audit_sha256=sha256(args.range_audit),
        legacy_checkpoint_rejection=legacy_rejected,
        support_refitting=False, coefficient_refitting=False, clipping=False,
        stage1_changed=False, stage2_changed=False, gpu_jobs_launched=False,
        stale_bank_normalizers_reused=False,
        limitations=[
            'Continuous physical coefficients are preserved; nearest shared-grid quantization adds the independently measured rounding error.',
            'Old stage-two coefficient embeddings and output labels require explicit migration or fresh training.',
            'Old BankDataset requires invalidated bank_log_normalizers; future physical training needs an explicit new loader.',
            'No FID or training improvement is established by this data conversion.'])
    report_path = output / 'conversion-report.json'
    write_json(report_path, report)
    files.append(report_path)
    manifest = {str(path.relative_to(output)): dict(sha256=sha256(path), bytes=path.stat().st_size)
                for path in files}
    manifest_path = output / 'prepared-manifest.json'
    write_json(manifest_path, manifest)
    files.append(manifest_path)
    if args.durable:
        print('Copying prepared caches and receipts to durable storage', flush=True)
        durable = args.durable.resolve()
        durable.mkdir(parents=True, exist_ok=True)
        copies = {str(path.relative_to(output)): checked_copy(path, durable/path.relative_to(output))
                  for path in files}
        receipt = dict(passed=True, copied_files=copies, all_sha256_verified=True,
                       created_unix=time.time(), local_output=str(output), durable_output=str(durable))
        write_json(output/'durable-copy.json', receipt)
        checked_copy(output/'durable-copy.json', durable/'durable-copy.json')
    print(json.dumps(dict(passed=True, output=str(output), durable=str(args.durable),
        coefficient_codec_identity=common_meta['coefficient_codec_identity'],
        caches={name: value['output_sha256'] for name, value in records.items()})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--source-audit', type=Path, required=True)
    parser.add_argument('--range-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--durable', type=Path)
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--legacy-checkpoint', type=Path)
    parser.add_argument('--recovery-helper', type=Path)
    parser.add_argument('--tokenizer', type=Path)
    args = parser.parse_args()
    if args.cpu_threads < 1 or bool(args.recovery_helper) != bool(args.tokenizer):
        parser.error('CPU threads must be positive; recovery helper and tokenizer must be supplied together')
    prepare(args)


if __name__ == '__main__':
    main()
