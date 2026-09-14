The queued Church stage-2 job now uses the released RQ-VAE `rqvae.metrics.fid.compute_fid` function for both FID4096 and FID50k. The subsequent compact-tokenizer reconstruction screen also uses the released file loader, Inception, dense NumPy statistics, and Frechet-distance function.

Generation writes float32 RGB pickles in the format expected by the repository. Rank 0 invokes the unmodified released evaluator on the complete sample population and broadcasts its result. The wrapper handles sampling, files, and synchronization; it performs no custom feature aggregation or FID arithmetic. The evaluator and Inception source hashes are checked against the recorded immutable upstream snapshot.

Each evaluation has a fresh directory. An existing `acts.npz` is rejected to prevent an old score from being reused. Generated codes, all Inception activations/statistics, and RGB-file hashes are retained. Temporary RGB pickles are removed only after evaluation succeeds. The RNG scope includes the evaluator's model initialization and data loader, preserving the training random sequence.

Validation completed before the launcher handoff:

- Full 4,096-image generation replay through the production adapter: **FID 16.7446212299**, versus the previous streaming result 16.7446241495; absolute difference **0.00000292**.
- CPU and GPU RNG states restored exactly on both ranks; model training mode restored; model and tokenizer weights unchanged.
- Stale activation cache rejected; all 4,096 Inception features retained; temporary RGB cleanup verified.
- Preparation smoke test completed using fresh latents, the compact codebook, calibration, and official reconstruction evaluation. Its 32-image reconstruction score is only a functional check, not a quality estimate.

The new supervisor adopted the running fine-tune at the existing process identities. Supervisor PID 117288 was replaced by 123179; torchrun PID 117290 and training worker PIDs 117300/117301 continued. The fine-tune entered its third epoch after the handoff. Its already-running epoch rFID monitor continues using the historical accumulator; the evaluator change applies to the subsequent tokenizer screen and stage-2 generation evaluations.

The new supervisor verifies all 361 files in its source manifest before each subsequent phase. It waits for all three fine-tuning epochs, prepares the new tokenizer artifacts, verifies the stage-2 preflight, and then starts fresh stage-2 training. Existing learning rates, optimizer settings, architecture, and sample settings are preserved.

Files: [FID integration](/workspace/Projects/laser/scripts/tools/church_official_fid.py), [stage-2 driver](/workspace/Projects/laser/scripts/tools/train_church_laser_three_epoch_official_fid.py), [preparation driver](/workspace/Projects/laser/scripts/tools/prepare_church_laser_three_epoch_official_fid.py), [supervisor](/workspace/Projects/laser/scripts/tools/run_church_laser_three_epoch_official_fid.py).

Evidence: [generation integration test](/workspace/Projects/laser/outputs/church-laser-three-epoch-20260914/official-fid-verification/verification.json), [preparation smoke result](/workspace/Projects/laser/outputs/church-laser-three-epoch-20260914/preparation-official-fid-smoke/reconstruction-screen.json), [completed supervisor handoff](/workspace/Projects/laser/outputs/church-laser-three-epoch-20260914/official-fid-handoff-complete.json), [source manifest](/workspace/Projects/laser/outputs/church-laser-three-epoch-20260914/official-fid-source-manifest.json).
