"""Collect offline teacher features through an isolated Omni inference worker."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from specforge.algorithms.dspark.capture import dspark_capture_features
from specforge.data.loss_mask import has_consecutive_supervised_tokens
from specforge.runtime.data_plane.capture_journal import FeatureJournal


class CaptureSample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: str
    source: str
    split: str
    inputs: dict[str, JsonValue]
    input_files: list[Path] = Field(default_factory=list)
    params: dict[str, JsonValue] = Field(default_factory=dict)


class WorkerReady(BaseModel):
    schema_version: Literal[1]
    provenance: dict[str, JsonValue]


class CaptureMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")
    generated_spans: list[tuple[int, int]] = Field(default_factory=list)
    layer_ids: list[int] = Field(default_factory=list)


class WorkerReply(BaseModel):
    request_id: str
    rejection: str | None
    metadata: CaptureMetadata


def collect_omni_features(
    *,
    worker_python: Path,
    worker_directory: Path,
    factory: str,
    worker_config: Path,
    inputs: Path,
    output: Path,
    limit: int,
    draft_config: Path,
    excluded_sources: tuple[str, ...] = (),
) -> None:
    if limit < 1:
        raise ValueError("Capture limit must be positive")
    samples = [
        CaptureSample.model_validate_json(line)
        for line in inputs.read_text().splitlines()
        if line.strip()
    ]
    if any(sample.source in excluded_sources for sample in samples):
        raise ValueError("Excluded source in capture input manifest")
    draft_configuration = json.loads(draft_config.read_text())
    expected_layer_ids = draft_configuration["dflash_config"]["target_layer_ids"]
    with tempfile.TemporaryDirectory(prefix="specforge-omni-") as temporary_directory:
        artifact_path = Path(temporary_directory) / "capture.pt"
        process = subprocess.Popen(
            [
                str(worker_python.absolute()),
                "-m",
                "sglang_omni.capture.worker",
                "--factory",
                factory,
                "--config",
                str(worker_config.resolve()),
                "--artifact-path",
                str(artifact_path),
            ],
            cwd=worker_directory,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            ready_line = process.stdout.readline()
            if not ready_line:
                raise RuntimeError(
                    f"Omni capture worker failed to start (exit {process.wait()})"
                )
            ready = WorkerReady.model_validate_json(ready_line)
            provenance = {
                "backend": ready.provenance,
                "factory": factory,
                "inputs_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
                "draft_config": draft_configuration,
                "strategy": "dspark",
            }
            journal = FeatureJournal(output, provenance)
            for record in journal.records:
                sample = samples[record["input_index"]]
                fingerprints = {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sample.input_files
                }
                if fingerprints != record["input_files_sha256"]:
                    raise ValueError(
                        f"Previously captured input changed: {sample.sample_id}"
                    )
            for input_index in range(len(journal.records), min(len(samples), limit)):
                sample = samples[input_index]
                request_id = str(input_index)
                process.stdin.write(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "inputs": sample.inputs,
                            "params": sample.params,
                        }
                    )
                    + "\n"
                )
                process.stdin.flush()
                reply_line = process.stdout.readline()
                if not reply_line:
                    raise RuntimeError(
                        f"Omni capture worker exited during input {input_index} (exit {process.wait()})"
                    )
                reply = WorkerReply.model_validate_json(reply_line)
                if reply.request_id != request_id:
                    raise ValueError("Capture response belongs to a different request")
                metadata = sample.model_dump(mode="json")
                metadata["input_files_sha256"] = {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sample.input_files
                }
                metadata["capture"] = reply.metadata.model_dump(mode="json")
                features = None
                rejection = reply.rejection
                if rejection is None:
                    if reply.metadata.layer_ids != expected_layer_ids:
                        raise ValueError(
                            "Captured layer order differs from the draft configuration"
                        )
                    tensors = torch.load(
                        artifact_path, map_location="cpu", weights_only=True
                    )
                    artifact_path.unlink()
                    features = dspark_capture_features(
                        tensors,
                        reply.metadata.generated_spans,
                        reply.metadata.layer_ids,
                    )
                    if (
                        features["target_last_hidden_states"].shape[1]
                        != draft_configuration["hidden_size"]
                    ):
                        raise ValueError(
                            "Captured hidden width differs from the draft configuration"
                        )
                    if not has_consecutive_supervised_tokens(features["loss_mask"]):
                        features = None
                        rejection = "no_supervised_anchor"
                    else:
                        metadata["input_ids"] = features["input_ids"].tolist()
                        metadata["shapes"] = {
                            name: list(tensor.shape)
                            for name, tensor in features.items()
                        }
                journal.append(metadata, features, rejection)
        finally:
            process.stdin.close()
            process.stdout.close()
            try:
                return_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise RuntimeError("Omni capture worker did not shut down")
        if return_code:
            raise RuntimeError(f"Omni capture worker exited with {return_code}")
