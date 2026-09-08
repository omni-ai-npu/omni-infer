# OpenPangu Omni multimodal calibration manifests

Use one entry per line. Blank lines and lines starting with `#` are ignored.
Paths may be absolute or relative to the manifest file.

Vision JSONL:

```jsonl
{"image": "/data/calib/images/000001.jpg"}
{"image": "/data/calib/images/000002.png"}
```

Audio JSONL:

```jsonl
{"audio": "/data/calib/audio/000001.wav"}
{"audio": "/data/calib/audio/000002.flac"}
```

A plain TXT file containing one path per line is also accepted. Audio is decoded
with `soundfile`, mixed to mono, and resampled to the checkpoint's 16 kHz rate.
Long audio is split into the same 30-second chunks used by serving.

Recommended minimum: 32 diverse images and 32 diverse audio clips. Keep the
distribution close to production data. Very short audio clips should be mixed
with medium/long clips so every strict GPTQ block sees at least 128 activation
rows; the quantizer raises instead of falling back to RTN when undersampled.
