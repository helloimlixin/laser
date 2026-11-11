"""Freeze the selected learned joint book, using beam16 integer encoding."""
from frozen_tokenizer import load_source as load_original,LASER_HASH,digest_state
from src.models.beam_joint_var import BeamJointQuantizer

def load_source(path,kind,device):
    model=load_original(path,kind,device)
    model.quantize=BeamJointQuantizer(model.quantize,beam_width=16,chunk_size=256,
        sequence_temperature=.01,max_relative_error=.01)
    assert model.quantize.vocab_size==4096 and model.quantize.levels==4
    return model.eval().requires_grad_(False)
