import json
import pytest
from src.tokenizer_fidelity import require_tokenizer_fidelity


def test_quality_gate_requires_matched_full_evaluation_and_exact_provenance(tmp_path):
    path=tmp_path/'quality.json'
    baseline=dict(checkpoint_sha256='checkpoint',codebook_sha256='codebook',images=50000,
        same_validation_images=True,same_reference_statistics=True,original_rfid=4.21,converted_rfid=4.30)
    path.write_text(json.dumps(baseline))
    assert require_tokenizer_fidelity(path,'checkpoint','codebook')['approved']
    accepted={**baseline,'original_rfid':4.215116853254415,'converted_rfid':4.362994621413748}
    path.write_text(json.dumps(accepted))
    with pytest.raises(ValueError):require_tokenizer_fidelity(path,'checkpoint','codebook',.1)
    approved=require_tokenizer_fidelity(path,'checkpoint','codebook',.15)
    assert approved['approved']
    path.write_text(json.dumps(approved))
    assert require_tokenizer_fidelity(path,'checkpoint','codebook',.15)==approved
    with pytest.raises(ValueError):require_tokenizer_fidelity(path,'checkpoint','codebook',.1)
    for update in [dict(converted_rfid=4.32),dict(images=4096),dict(same_validation_images=False),
                   dict(same_reference_statistics=False),dict(checkpoint_sha256='other'),
                   dict(codebook_sha256='other'),dict(converted_rfid=float('nan')),
                   dict(original_rfid=6.4,converted_rfid=6.45)]:
        path.write_text(json.dumps({**baseline,**update}))
        with pytest.raises(ValueError):require_tokenizer_fidelity(path,'checkpoint','codebook')
