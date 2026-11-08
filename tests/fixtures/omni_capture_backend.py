"""Synthetic inference adapter for exercising the real capture subprocess protocol."""

import torch
from pydantic import JsonValue
from sglang_omni.capture.backend import CaptureInput, CaptureReply
from sglang_omni.capture.features import CaptureRequest, TokenSpan, capture_sequences


class FixtureBackend:
    provenance: dict[str, JsonValue] = {"implementation": "transport-fixture"}

    def capture(self, request: CaptureInput) -> CaptureReply:
        if request.inputs.get("length_limited"):
            return CaptureReply(sequence=None, metadata={}, rejection="length")
        token_ids = request.inputs["token_ids"]
        capture = CaptureRequest(
            input_ids=torch.tensor(token_ids),
            generated_spans=tuple(
                TokenSpan(start_token_index=start, end_token_index=stop)
                for start, stop in request.inputs["generated_spans"]
            ),
            layer_ids=(0, 2),
        )
        final_states = torch.tensor(token_ids, dtype=torch.float32)[:, None].repeat(
            1, 3
        )
        capture_sequences(
            [capture], [len(token_ids)], [0], final_states.repeat(1, 2), final_states
        )
        return CaptureReply(sequence=capture.result, metadata={})

    def close(self) -> None:
        pass


def create_backend(configuration: dict[str, JsonValue]) -> FixtureBackend:
    return FixtureBackend()
