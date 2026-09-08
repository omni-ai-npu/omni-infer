# Default calibration data

`wikitext_train.parquet` — the WikiText-2 train split (a `text` column), the
default calibration set. Bundled so the example/smoke commands run out of the
box. ~5.9 MB.

It's a generic-text calib; for tasks whose distribution differs (agent rollouts,
CoT), a matching calibration set gives better quantization (GPTQ is most sensitive
to calibration distribution). Pass your own via `--calib-data`.

Source: WikiText-2 (Merity et al., 2016), publicly available.
