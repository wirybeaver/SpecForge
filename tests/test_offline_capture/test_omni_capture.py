"""Cross-environment capture, DSpark adaptation and restart integration."""

import json
import sys
from pathlib import Path

import pytest
import torch

from specforge.algorithms.common.hidden_states_data import (
    build_dspark_collator,
    build_dspark_offline_reader,
    normalize_dspark_offline_sample,
)
from specforge.inference.omni_capture import collect_omni_features
from specforge.runtime.data_plane.feature_store import load_feature_file

pytest.importorskip("sglang_omni")


def test_capture_subprocess_reader_and_resume(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.jsonl"
    rows = [
        {
            "sample_id": "one",
            "source": "fixture",
            "split": "train",
            "inputs": {
                "token_ids": [8, 9, 10, 11, 12, 13],
                "generated_spans": [[1, 3], [4, 6]],
            },
        },
        {
            "sample_id": "rejected",
            "source": "fixture",
            "split": "train",
            "inputs": {"length_limited": True},
        },
        {
            "sample_id": "two",
            "source": "fixture",
            "split": "train",
            "inputs": {"token_ids": [20, 21, 22, 23], "generated_spans": [[2, 4]]},
        },
    ]
    inputs.write_text("".join(json.dumps(row) + "\n" for row in rows))
    worker_config = tmp_path / "worker.json"
    worker_config.write_text("{}")
    draft_config = tmp_path / "draft.json"
    draft_config.write_text(
        json.dumps({"hidden_size": 3, "dflash_config": {"target_layer_ids": [0, 2]}})
    )
    output = tmp_path / "features"
    options = dict(
        worker_python=Path(sys.executable),
        worker_directory=Path(__file__).resolve().parents[2],
        factory="tests.fixtures.omni_capture_backend:create_backend",
        worker_config=worker_config,
        inputs=inputs,
        output=output,
        draft_config=draft_config,
    )
    collect_omni_features(**options, limit=2)
    manifest = json.loads((output / "manifest.json").read_text())
    assert [record["feature_file"] for record in manifest["records"]] == [
        "00000000.ckpt",
        None,
    ]
    torch.save({"incomplete": True}, output / "00000001.ckpt")
    collect_omni_features(**options, limit=3)
    manifest = json.loads((output / "manifest.json").read_text())
    assert [record["sample_id"] for record in manifest["records"]] == [
        "one",
        "rejected",
        "two",
    ]
    assert manifest["records"][1]["rejection"] == "length"
    reader = build_dspark_offline_reader(
        "dspark", str(output), run_id="test", ttt_length=8, max_len=32
    )
    batch = build_dspark_collator()(
        [
            normalize_dspark_offline_sample(
                load_feature_file(reference.feature_store_uri.removeprefix("file://")),
                32,
            )
            for reference in reader
        ]
    )
    assert batch["input_ids"].tolist() == [
        [8, 9, 10, 11, 12, 13],
        [20, 21, 22, 23, 0, 0],
    ]
    assert batch["loss_mask"].tolist() == [[0, 1, 1, 0, 1, 1], [0, 0, 1, 1, 0, 0]]
    assert batch["target_last_hidden_states"][1, :4, 0].tolist() == [20, 21, 22, 23]
    assert batch["hidden_states"].shape == (2, 6, 6)
    draft_config.write_text(
        json.dumps({"hidden_size": 3, "dflash_config": {"target_layer_ids": [1, 2]}})
    )
    with pytest.raises(ValueError, match="provenance"):
        collect_omni_features(**options, limit=3)
