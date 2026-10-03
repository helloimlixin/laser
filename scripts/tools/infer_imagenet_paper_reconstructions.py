#!/usr/bin/env python3
"""CPU FP32 reconstruction of fixed validation examples for paper figures."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/rq-vae-transformer"))
from src.models.rqvae.rqvae import RQVAE
from src.models.dictionary_learner import DictionaryLearning

SHAS = {
    "laser4": "dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab",
    "laser2": "b064c9aed64224e85e91a0b17d231cbc6a6a0af994c7caf371f9c5a2b31a1576",
    "rq4": "e8eb312e6c0dbd8e490ca95f69e3c3b417162625ff9320bbfff791cbf832b694",
    "vqgan16": "845a68805098cb666420d5db93df53f3a3b6dd443e6dd85c05759c5b998cd663",
}


def load_model(key, assets, taming):
    checkpoint = (
        Path("/mnt/laser-stage2-7h100/assets/best_rfid_slot1_model.pt") if key == "laser4"
        else assets / "laser2/best_rfid_slot1_model.pt" if key == "laser2"
        else assets / "rq4/model.pt" if key == "rq4"
        else assets / "vqgan16/last.ckpt"
    )
    with checkpoint.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    assert digest == SHAS[key], (key, digest)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    if key == "vqgan16":
        sys.path.insert(0, str(taming))
        shim = types.ModuleType("torch._six")
        shim.string_classes = (str, bytes)
        sys.modules.setdefault("torch._six", shim)
        from taming.models.vqgan import VQModel
        config = OmegaConf.load(assets / "vqgan16/model.yaml")
        params = OmegaConf.to_container(config.model.params, resolve=True)
        params["lossconfig"] = {"target": "torch.nn.Identity"}
        model = VQModel(**params)
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not missing and all(k.startswith("loss.") for k in unexpected), (missing, unexpected)
    else:
        config = (json.loads((assets / f"{key}-config.json").read_text()) if key.startswith("laser")
                  else OmegaConf.to_container(OmegaConf.load(assets / "rq4/config.yaml"), resolve=True))
        arch = config["arch"]
        hp = arch["hparams"]
        kwargs = dict(hp)  # RQVAE accepts latent/code shapes through **kwargs.
        kwargs["bottleneck_type"] = "rq"
        model = RQVAE(**kwargs, ddconfig=arch["ddconfig"], checkpointing=False)
        if key.startswith("laser"):
            kwargs = {k: v for k, v in hp.items() if k in inspect.signature(DictionaryLearning).parameters}
            kwargs.update(num_embeddings=hp["n_embed"], embedding_dim=hp["embed_dim"])
            model.quantizer = DictionaryLearning(**kwargs)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Later versions add training-only dictionary accounting buffers.
        assert not unexpected and all(k.startswith("quantizer._") for k in missing), (missing, unexpected)
    return model.eval().requires_grad_(False), digest


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(SHAS), required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--taming", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model, digest = load_model(args.model, args.assets, args.taming)
    print(f"Loaded {args.model}, verified SHA-256 {digest}", flush=True)
    selection = json.loads((args.work / "selection.json").read_text())
    records = selection["samples"][:args.limit or None]
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    output = args.work / "pixels"
    output.mkdir(exist_ok=True)
    validation = []
    archived = args.work.parent / "source/figures"
    old_refs = Image.open(archived / "reference-images.png").convert("RGB")
    old_recons = Image.open(archived / "reconstruction-grid.png").convert("RGB")
    row = ["vqgan16", "rq4", "laser2", "laser4"].index(args.model)
    for sample in records:
        index = sample["index"]
        target = output / f"{index:02d}-{args.model}.png"
        if target.exists():
            continue
        started = time.monotonic()
        x = transform(Image.open(args.work / "inputs" / sample["source_file"]).convert("RGB")).unsqueeze(0)
        ref = x.mul(0.5).add(0.5).clamp(0, 1).squeeze(0).permute(1,2,0).numpy()
        ref = np.rint(ref*255).astype(np.uint8)
        if args.model == "laser4":
            Image.fromarray(ref).save(output / f"{index:02d}-reference.png")
        if args.model.startswith("laser"):
            z = model.encode(x).permute(0,3,1,2).contiguous()
            quantized, _, _ = model.quantizer(z)
            recon = model.decode(quantized.permute(0,2,3,1).contiguous())
        else:
            recon = model(x)[0]
        assert torch.isfinite(recon).all()
        rec = recon.mul(0.5).add(0.5).clamp(0,1).squeeze(0).permute(1,2,0).numpy()
        rec = np.rint(rec*255).astype(np.uint8)
        Image.fromarray(rec).save(target)
        if index <= 8:
            box = ((index-1)*256,0,index*256,256)
            assert np.array_equal(ref, np.asarray(old_refs.crop(box))), f"Reference preprocessing mismatch {index}"
            prior = np.asarray(old_recons.crop((box[0],row*256,box[2],(row+1)*256)))
            error = np.abs(rec.astype(np.float32)-prior.astype(np.float32))
            validation.append(dict(index=index,mean_absolute_uint8_difference=float(error.mean()),max_uint8_difference=float(error.max())))
        print(json.dumps(dict(model=args.model,index=index,seconds=round(time.monotonic()-started,2),validation=validation[-1] if index<=8 else None)),flush=True)
    result = dict(model=args.model,checkpoint_sha256=digest,device="cpu",precision="float32",count=len(records),archive_comparison=validation)
    (args.work / f"inference-{args.model}.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
