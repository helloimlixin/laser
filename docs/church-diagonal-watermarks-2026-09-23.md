The thin diagonal crosses visible in several Church samples coincide with
generated stock-photo watermark lettering. In the step-15,000 preview, row 5,
column 2 is a clear example. This pattern is already present in the original
LSUN training photographs, before encoding, token caching, or generation.

Training image index 89,267, LMDB key
`b472369633b670b8710a3e5c8203602f2cf21c83`, contains diagonal cross lines,
Shutterstock lettering, and a stock-photo footer. Its original 256×401 JPEG
was extracted byte-for-byte from the read-only training LMDB; SHA-256 is
`5d4f93d7638486ce5f6f744a74ac546ac4d05a096052954f624de145d3fb285e`.
Its previously saved continuous-cache reconstruction retains those lines and
lettering. The cache probe used the same sorted LMDB index order. Another
original, index 14,113, contains a photographer website footer.

This evidence supports learned training-data watermarks as the explanation
for these particular crosses. It does not diagnose every possible diagonal
pattern or the separate dark coefficient artifacts. The fused training
attention is not needed to produce the pattern: it exists in real inputs and
their frozen-tokenizer reconstructions.

Restarting training on the same population would retain this source of
watermarks. A remedy would exclude verified watermarked examples from the
stage-2 training population, reuse the corresponding clean cache entries,
and evaluate a separately tracked fine-tune. No dataset filtering, sampling
change, checkpoint replacement, or restart was made during this inspection.
Filtering has not yet been implemented or evaluated, and the prevalence of
watermarks in the full dataset has not been measured.

Original image bytes, provenance, extraction and comparison scripts, and a
three-panel evidence figure are stored in the current run's
`diagonal-debug/` directory. The generated panel is an unrelated example,
not a reconstruction of the displayed original photograph.

The reconstruction panel is only a decoded visualization of training tokens;
it is not used as source imagery for training. A subsequent
[original-image source audit](church-original-image-source-audit-2026-09-23.md)
freshly encoded 24 original LMDB photographs and matched every cached atom ID.
Re-encoding the reconstruction images failed that comparison, confirming the
distinction between the actual source photographs and diagnostic outputs.
