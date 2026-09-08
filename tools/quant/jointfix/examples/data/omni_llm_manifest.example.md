# Omni LLM calibration manifest

Use one JSON object per line.  Media paths may be absolute or relative to the
manifest.  `prompt` is accepted as an alias of `text`.

```jsonl
{"image":"images/chart.jpg","text":"请分析图表并给出结论。"}
{"audio":"audio/question.wav","text":"请理解音频并回答问题。"}
{"image":"images/page.png","audio":"audio/instruction.wav","text":"请根据图片和音频完成任务。"}
```

The first implementation supports at most one image and one audio file in each
request.  If `text` is omitted, JointFix supplies a generic modality-matching
question.  Plain image/audio path lines are also accepted.

Run decoder calibration with real BF16 tower/projector outputs by adding:

```bash
--calib-data /data/calib/omni_llm.jsonl \
--calib-format omni \
--n-samples 32 --seq-len 1024
```

`--seq-len` is an upper bound for each expanded multimodal prompt.  Requests
remain variable length and are not padded into the activation statistics.
