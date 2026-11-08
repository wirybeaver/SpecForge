# Omni teacher capture

`specforge capture` owns dataset iteration, durable feature publication, rejection records and resume. It talks to an Omni inference worker in a separate Python environment. The worker performs model-specific generation and replay; SpecForge converts the returned generic token-aligned features into its DSpark training contract. The transport and journal contain no Ming token IDs, audio preprocessing, hidden width or capture-layer constants.

## Capture

Prepare separate uv environments for Omni and SpecForge. Configure a compatible GPU capture runtime in the Omni environment; installing the normal Omni dependency pin alone does not establish compatibility. The first implemented model adapter is Ming ASR. Its inspected SGLang candidate revision is recorded in `examples/configs/capture/ming-3.0-asr-tiny.json`; GPU compatibility and numerical validation are still pending.

Use a pinned local Ming checkpoint at revision `ac95e7b1678c08ea6f4a2581a051e432ed63e9b7`. Set its path in the worker configuration. Relative model and audio paths resolve in the worker directory; absolute paths are preferable when the repositories differ. Source revision strings are provenance declarations and must match the installed code and checkpoint.

The JSONL input format is model-independent. Each row carries sample identity, source, split, opaque model inputs and optional parameters. List local assets in `input_files` so their content is checked on resume. For Ming:

```json
{"sample_id":"en-001","source":"approved","split":"train","input_files":["/path/to/en.wav"],"inputs":{"audio_path":"/path/to/en.wav"},"params":{}}
{"sample_id":"zh-001","source":"approved","split":"train","input_files":["/path/to/zh.wav"],"inputs":{"audio_path":"/path/to/zh.wav"},"params":{"language":"zh"}}
```

From the SpecForge environment:

```bash
specforge capture \
  --worker-python /path/to/omni-venv/bin/python \
  --worker-directory /path/to/sglang-omni \
  --factory sglang_omni.models.ming3_asr.capture:create_capture_backend \
  --worker-config examples/configs/capture/ming-3.0-asr-tiny.json \
  --draft-config configs/ming-3.0-asr-tiny-dspark.json \
  --inputs audio.jsonl --output cache/ming-asr --limit 2 \
  --exclude-source Emilia-YODAS --exclude-source 'Common Voice 17'
```

Source exclusions match the exact source names in the input manifest. Those two corpora must remain excluded from the Ming experiment. The limit counts processed inputs including previous runs, not accepted samples. Use a dedicated output directory and one writer. Run training only after capture has stopped; the trainer scans feature files rather than the capture journal.

The worker first announces schema version and provenance, then responds to one request at a time. Tensors travel through a caller-owned temporary artifact on the shared local filesystem. SpecForge checks request identity, layer order, hidden width, complete rows and generated spans before constructing a mask. Prompt/context rows remain present with zero loss. Samples without two consecutive supervised positions are rejected by the DSpark adapter. This process adapter is local; it does not imply support for remote paths or HTTP capture.

Feature files use the existing `.ckpt` dictionary schema. Publication writes the tensor file before atomically committing its journal entry. Recovery distinguishes processed-input count from accepted-file count and removes the single uncommitted next file after an interrupted publication. Unexpected worker failures stop capture. Resume checks input manifest, draft configuration, backend provenance and declared local asset hashes.

## Train and extend

The Ming draft configuration uses full vocabulary, five MHA layers and explicit mask/EOS ID 156892. The decoder output layer IDs are 1, 6, 13, 17 and 20; final states must be post-norm. The pinned checkpoint uses `model.word_embeddings.weight` and `lm_head.weight`. After GPU feature validation:

```bash
specforge train --config examples/configs/offline/colocated/ming-3.0-asr-tiny-dspark.yaml
```

The recipe uses one epoch, one sample per batch and 16 anchors for a bounded smoke dataset. Existing training resume and export commands remain unchanged. Train/save/resume/export, numerical parity and speculative serving have not been validated on GPU.

A new model supplies an Omni capture factory implementing the same contract and a compatible draft recipe. It reuses this subprocess adapter and journal. A new training algorithm supplies its feature conversion and eligibility policy; the current command intentionally selects DSpark. Online delivery can later replace the local artifact transport while retaining the same token/layer semantics.

## CPU integration check

With both source trees available, run the contract test in a small CPU uv environment from SpecForge:

```bash
PYTHONPATH=/path/to/SpecForge:/path/to/sglang-omni uv run --no-project \
  --with torch --with numpy --with pydantic --with pytest --with click --with pyyaml \
  python -m pytest tests/test_offline_capture/test_omni_capture.py -q
```

It exercises the actual worker subprocess, a synthetic model adapter, variable token lengths and disjoint generated spans, rejection/restart, and the native reader/normalizer/collator. Synthetic tensors verify transport and ownership only; they provide no evidence of model numerical parity.
