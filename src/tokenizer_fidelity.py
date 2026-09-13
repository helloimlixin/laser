"""Require matched full-validation evidence before approving a stage-2 tokenizer."""
import json
import math
from pathlib import Path


def require_tokenizer_fidelity(report_path, checkpoint_sha256, codebook_sha256, max_drift=.1):
    report=json.loads(Path(report_path).read_text())
    if not math.isfinite(max_drift) or max_drift<0:
        raise ValueError('RFID drift tolerance must be finite and nonnegative')
    if report.get('checkpoint_sha256')!=checkpoint_sha256:
        raise ValueError('Fidelity report belongs to a different source checkpoint')
    if report.get('codebook_sha256')!=codebook_sha256:
        raise ValueError('Fidelity report belongs to a different integer codebook')
    if report.get('images')!=50000 or not report.get('same_validation_images'):
        raise ValueError('Full matched 50,000-image validation is required')
    if not report.get('same_reference_statistics'):
        raise ValueError('Original and converted tokenizer must use the same FID reference')
    original=float(report['original_rfid'])
    converted=float(report['converted_rfid'])
    if not all(math.isfinite(x) and x>=0 for x in (original,converted)):
        raise ValueError('Nonfinite or negative reconstruction FID')
    source=float(report.get('source_reported_rfid',4.210914134979248))
    if not math.isfinite(source) or abs(original-source)>.02:
        raise ValueError('Matched baseline does not reproduce the source 4.21 rFID within 0.02')
    drift=converted-original
    if drift>max_drift:
        raise ValueError(f'Tokenizer rFID drift {drift:.6f} exceeds {max_drift:.6f}: {original:.6f} -> {converted:.6f}')
    return {**report,'measured_drift':drift,'max_allowed_drift':max_drift,'approved':True}
