"""Atomic offline capture journal with separate input and accepted-sample cursors."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from pydantic import JsonValue

SCHEMA_VERSION = 1


class FeatureJournal:
    """Commit one processed input at a time; manifest publication is the commit point."""

    def __init__(self, directory: Path, provenance: dict[str, JsonValue]) -> None:
        self.directory: Path = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path: Path = directory / "manifest.json"
        self.provenance: dict[str, JsonValue] = provenance
        self.records: list[dict[str, JsonValue]] = []
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
            if (
                manifest["schema_version"] != SCHEMA_VERSION
                or manifest["provenance"] != provenance
            ):
                raise ValueError(
                    "Capture provenance changed; use a new output directory"
                )
            else:
                self.records = manifest["records"]
        elif any(directory.glob("*.ckpt")):
            raise ValueError("Output contains feature files without a capture manifest")
        else:
            self.publish_records([])
        self.accepted_count: int = sum(
            record["feature_file"] is not None for record in self.records
        )
        expected_files = {
            record["feature_file"]
            for record in self.records
            if record["feature_file"] is not None
        }
        for filename in expected_files:
            if not (directory / filename).is_file():
                raise FileNotFoundError(directory / filename)
            else:
                pass
        # note (Codex): A crash after tensor publication leaves an uncommitted file invisible to the journal.
        for feature_path in directory.glob("*.ckpt"):
            if feature_path.name in expected_files:
                continue
            elif feature_path.name == f"{self.accepted_count:08d}.ckpt":
                feature_path.unlink()
            else:
                raise ValueError(f"Unexpected feature file: {feature_path}")

    def append(
        self,
        metadata: dict[str, JsonValue],
        features: dict[str, torch.Tensor] | None,
        rejection: str | None,
    ) -> None:
        assert (features is None) == (rejection is not None)
        filename = None
        if features is not None:
            filename = f"{self.accepted_count:08d}.ckpt"
            temporary_path = self.directory / (filename + ".tmp")
            with temporary_path.open("wb") as output:
                torch.save(features, output)
                output.flush()
                os.fsync(output.fileno())
            temporary_path.replace(self.directory / filename)
        else:
            pass
        record = dict(
            metadata,
            input_index=len(self.records),
            feature_file=filename,
            rejection=rejection,
        )
        self.publish_records([*self.records, record])
        self.records.append(record)
        self.accepted_count += features is not None

    def publish_records(self, records: list[dict[str, JsonValue]]) -> None:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "provenance": self.provenance,
            "records": records,
        }
        temporary_path = self.manifest_path.with_suffix(".json.tmp")
        with temporary_path.open("w") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        temporary_path.replace(self.manifest_path)
