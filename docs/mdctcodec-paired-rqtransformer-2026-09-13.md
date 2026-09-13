# Controlled LASER/RVQ audio prior experiment

The new experiment trains two fresh text-to-speech priors at a nominal 6 kbps. It addresses the missing RVQ stage2 control. It does not use the pretrained F5 or Chatterbox scores to judge which bottleneck is better.

## Integer representation and architecture

The existing LASER prior already predicts discrete atom and coefficient fields. Its within-frame decoder was a GRU. The new prior uses an eight-layer temporal transformer and a two-layer causal depth transformer, with a three-layer phoneme encoder, width512 and eight attention heads. This follows the temporal/spatial-plus-depth factorization of [RQ-Transformer](https://arxiv.org/abs/2203.01941), adapted to speech with text cross-attention and learned speaker IDs. It is an audio adaptation, not the authors' unmodified image model.

Each 150 Hz frame has four categorical fields:

| Codec | Fields | Fixed-width payload |
|---|---|---|
| LASER | atom1, coefficient1+63, atom2, coefficient2+63 | 13+7+13+7 =40 bits |
| RVQ | codebook1 ID, codebook2 ID, codebook3 ID, codebook4 ID | 10+10+10+10 =40 bits |

LASER retains8192 atoms and127 signed coefficient levels. Coefficients are quantized values from the frozen stage1 codec, not arbitrary floats cast to integers. One sparse pair can equivalently be represented by the dense integer `atom_id *127 + coefficient_code`, giving1,040,384 legal joint values. The new model factorizes their prediction into smaller conditional heads. This does not enlarge the stage1 dictionary or change bitrate. EOS belongs only to the first prediction head and is excluded from payloads. A repeated LASER atom is masked within a frame; repeated integer IDs across different RVQ codebooks are legal.

The depth transformer sees only the temporal context and preceding fields. Training and cached inference use the same causal factorization and cumulative field embeddings. Legacy GRU checkpoints still load through the default configuration.

## Frozen comparison

The plan is `outputs/mdctcodec_tts_paired/plan.json`. Final cache hashes and runnable training configurations are produced in `protocol.json`, `laser.yaml`, and `rvq.yaml` after GPU cache encoding and alignment verification.

Both codecs come from the completed paired200,000-generator-update stage1 experiment. Each was selected by the same validation ViSQOL rule within that budget. The LASER codec is the one used by the previous stage2 run; its SHA begins `3f4d31216ff5`. The RVQ codec SHA begins `c324e3a06b92`. The later400k codec continuation is a separate experiment.

Both fresh priors use seed20260913, the same phoneme/speaker mappings, all38,288 full training utterances, identical transcript splits, frame-budget8192 batches, accumulation4, and160 epochs. The LR schedule uses the exact number of batches across all160 epochs:1,000-update warmup to3e-4, then cosine decay to1e-5. Guided alignment weight0.2 decays over8,000 updates. Per-update RNG reset prevents codec or ASR initialization from changing the shared training dropout stream.

All49,584,128 common parameters—including phoneme/speaker embeddings, text, temporal and depth transformers—are copied exactly at initialization and verified by SHA256. LASER has66,639,103 total prior parameters; RVQ has53,783,553 because its vocabulary embeddings and output heads are smaller. This is a matched-backbone, matched-update comparison, not an equal-total-parameter or equal-FLOP comparison. Vocabulary NLLs are monitored within each arm and are not compared as a quality ranking.

Training records an auditable hash chain of actual utterance indices, phonemes, speakers, lengths and batch order at every optimizer update, together with the LR and cumulative frame count. Reporting refuses incomplete epoch budgets or divergent audits. Resume restores optimizer state and removes audit rows beyond the restored checkpoint.

The initial GPU preflight found approximately1% different RVQ IDs between CPU and CUDA encoding on eight full utterances. CPU shards are diagnostic only. The final RVQ cache is therefore encoded on CUDA with the LASER cache's FP32 policy: matmul TF32 disabled, cuDNN TF32 enabled. Serialized decoding was checked, and both payloads occupy exactly five bytes per frame. The original length and codec metadata remain outside the raw6 kbps payload.

## Selection, evaluation and logging

Primary selection uses the lowest Whisper-large-v3 WER on64 fixed validation-only prompts, evaluated every10 completed epochs; ties select the earlier checkpoint. A secondary comparison uses each fixed epoch160 endpoint. The two arms use the same prompt selection, generation settings and scoring implementation. Selection never uses test results.

The new frozen test manifest contains100 known speakers and100 distinct normalized target texts, excluding every normalized target text in the earlier100-clip benchmark. All cached0.5–12-second utterances were eligible. Stage2 train, validation and test texts are disjoint. The codecs may have seen these target recordings; this is not an unseen-speaker or completely unseen-audio claim.

The queued evaluation includes reference audio, reconstruction from each frozen codec, both validation-selected priors and both endpoints. It measures corpus WER/CER, UTMOS, ECAPA speaker similarity, serialized payload rate and synthesis speed, with paired speaker-bootstrap confidence intervals. Each generation run uses one GPU without the other prior running. UTMOS is not human MOS, and these results are not a general TTS SOTA benchmark.

W&B receives latest optimizer checkpoints and the best three within-arm NLL and validation-WER checkpoints every five completed epochs, plus16 speech previews, reference audio, waveforms, log-mel spectrograms and blue attention maps. Stage1 keeps its existing separate latest/top-three-ViSQOL upload policy.

## Execution and verification

`scripts/tools/run_mdctcodec_tts_pair.py` queues production GPU caching, cache alignment verification, full trainer smoke runs, both160-epoch training runs and the dependent benchmark. It waits for the existing codec campaign to exit and for both GPUs to be idle. It does not pause running codec trainers. The new stage2 campaign has its own24 assigned-GPU-hour ceiling, including GPU preparation, validation and evaluation; it does not alter the earlier campaign's ceiling. If the budget expires, checkpoints are saved and no completed matched result is published.

The model GPU preflight passed on an H200: two optimizer updates per arm, four full-utterance microbatches per update, finite gradients through text and depth attention, and serialized generation/decode. These are untrained smoke samples and their timings are not performance benchmarks. Thirty-five CPU tests passed for legacy compatibility, depth causality, cached inference, token/EOS constraints, shared initialization, data audit mismatch rejection, fresh test selection, existing evaluation behavior and audio media. Full trainer smoke runs are an additional queue prerequisite before production training.
