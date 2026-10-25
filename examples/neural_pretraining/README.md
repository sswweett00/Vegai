# Neural pretraining smoke fixture

`mini_turkish_byte_bpe.txt` is a repository-authored, MIT-licensed UTF-8 fixture.
It has eight short line-framed Turkish records and is intentionally too small to
serve as a training, held-out evaluation, or language-quality corpus.

Use it to exercise the native byte-BPE merge learner, teacher-forced
output-adapter checkpoint path, and calibrated causal generation wiring:

```bash
./build/vegai_neural_pretrain tokenizer-train \
  examples/neural_pretraining/mini_turkish_byte_bpe.txt /tmp/fixture.bpe 300
./build/vegai_neural_pretrain train \
  examples/neural_pretraining/mini_turkish_byte_bpe.txt /tmp/fixture.vgnr /tmp/fixture.bpe 1
# Or persist/reload a bounded continuous chunk; source-order identity is an
# operator provenance responsibility.
./build/vegai_neural_pretrain train-chunk \
  examples/neural_pretraining/mini_turkish_byte_bpe.txt /tmp/fixture.vgnr /tmp/fixture.bpe 4
./build/vegai_neural_pretrain generate /tmp/fixture.vgnr /tmp/fixture.bpe "Trabzon" 16 7
```

The immutable byte count and SHA-256 are in `dataset.json`. Regenerate or
replace neither without updating that manifest and independent provenance
records. See [`docs/NEURAL_CAUSAL_SYSTEM.md`](../../docs/NEURAL_CAUSAL_SYSTEM.md)
for the trainer’s exact scope and assurance boundaries.
