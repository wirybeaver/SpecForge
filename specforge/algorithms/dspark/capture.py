"""Adapt neutral teacher outputs to the offline DSpark feature contract."""

import torch


def dspark_capture_features(
    tensors: dict[str, torch.Tensor],
    generated_spans: list[tuple[int, int]],
    layer_ids: list[int],
) -> dict[str, torch.Tensor]:
    input_ids = tensors["input_ids"]
    auxiliary_states = tensors["auxiliary_hidden_states"]
    final_states = tensors["final_hidden_states"]
    if input_ids.ndim != 1 or input_ids.dtype != torch.int64:
        raise ValueError("Capture input_ids must be int64[L]")
    if final_states.ndim != 2 or final_states.shape[0] != len(input_ids):
        raise ValueError(
            "Capture must include a final hidden row for every input token"
        )
    if auxiliary_states.shape != (
        len(input_ids),
        len(layer_ids) * final_states.shape[1],
    ):
        raise ValueError("Capture auxiliary rows or layer widths do not match")
    loss_mask = torch.zeros_like(input_ids)
    for start, stop in generated_spans:
        if not 0 <= start < stop <= len(input_ids):
            raise ValueError("Generated token span is outside the captured sequence")
        loss_mask[start:stop] = 1
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "hidden_states": auxiliary_states.to(torch.bfloat16),
        "target_last_hidden_states": final_states.to(torch.bfloat16),
    }
