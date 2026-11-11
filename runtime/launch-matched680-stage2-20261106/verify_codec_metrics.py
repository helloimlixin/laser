import sys,json,time
from pathlib import Path
import torch,numpy as np
base=Path('/tmp/laser-var-stage2-matched680-20261106');sys.path.insert(0,str(base/'runtime'))
from frozen_tokenizer import load_source,digest_state
from src.rqvae_metrics import OriginalRQVAEInception,DistributedOriginalRQVAEMetrics,load_reference_statistics
cfg=json.loads((base/'config.json').read_text());torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
source=load_source(cfg['checkpoint'],'laser','cuda:0');before=digest_state(source)
cache=torch.load(cfg['smoke_cache'],map_location='cpu',weights_only=True)
with torch.no_grad():
 with torch.autocast('cuda',dtype=torch.bfloat16):z=source.quant_conv(source.encoder(cache['images'][:2].cuda()))
 code=source.quantize.decompose(z);decoded,inputs=source.quantize.from_codes(code['codes'])
 torch.testing.assert_close(decoded,code['latent'],atol=0,rtol=0);torch.testing.assert_close(inputs,code['inputs'],atol=0,rtol=0)
 assert code['codes'].shape==(2,340,2) and code['inputs'].shape==(2,336,32)
 with torch.autocast('cuda',dtype=torch.bfloat16):images=source.decoder(source.post_quant_conv(decoded)).float().clamp(-1,1)
 assert images.isfinite().all();assert digest_state(source)==before
 inception=OriginalRQVAEInception().eval().requires_grad_(False).cuda()
 features,logits=inception((images+1)*.5,return_logits=True)
 assert features.shape==(2,2048) and logits.shape==(2,1008) and features.dtype==torch.float32
 assert features.isfinite().all() and logits.isfinite().all()
 metric=DistributedOriginalRQVAEMetrics('cuda:0',compute_inception_score=True,reference_stats_path=cfg['fid_reference'],inception=inception)
 metric.update((images+1)*.5,real=False)
 assert int(metric.fake_count)==2 and metric.fake_probabilities[0].dtype==torch.float32
 mu,sigma=load_reference_statistics(cfg['fid_reference'])
 assert mu.shape==(2048,) and sigma.shape==(2048,2048)
report=dict(passed=True,trained_codec_integer_roundtrip_bit_exact=True,teacher_forcing_inputs_bit_exact=True,tokenizer_digest_unchanged=True,tokens_per_image=680,vocabulary=4096,generated_pixels_finite=True,inception_fp32=True,inception_features=2048,inception_logits=1008,metric_update_passed=True,full_evaluation_run=False,reference_valid=True,time=time.time())
(base/'codec-metric-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
