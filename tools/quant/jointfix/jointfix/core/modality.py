# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Modality-aware calibration containers and MoE routing statistics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class CalibrationInputs:
    """Variable-length decoder inputs and aligned text/non-text token masks."""

    hidden: List[torch.Tensor]
    token_is_text: List[torch.Tensor]


class MoERouteStats:
    """Compact per-layer storage for MDMixQ SGFR computed after collection."""

    def __init__(self, num_experts: int, routed_scaling_factor: float = 1.0):
        self.num_experts = int(num_experts)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.topk_ids: List[torch.Tensor] = []
        self.topk_weights: List[torch.Tensor] = []
        self.selected_logits: List[torch.Tensor] = []
        self.token_is_text: List[torch.Tensor] = []

    def update(self, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
               selected_logits: torch.Tensor, token_is_text: torch.Tensor) -> None:
        rows = topk_ids.shape[0]
        token_is_text = token_is_text.detach().reshape(-1).bool()
        if token_is_text.numel() != rows:
            raise ValueError(
                f"route modality rows {token_is_text.numel()} != routed rows {rows}"
            )
        self.topk_ids.append(topk_ids.detach().to(dtype=torch.int32, device="cpu"))
        self.topk_weights.append(topk_weights.detach().float().cpu())
        self.selected_logits.append(selected_logits.detach().float().cpu())
        self.token_is_text.append(token_is_text.cpu())

    def merge(self, other: "MoERouteStats") -> None:
        if self.num_experts != other.num_experts:
            raise ValueError("cannot merge route stats with different expert counts")
        self.topk_ids.extend(other.topk_ids)
        self.topk_weights.extend(other.topk_weights)
        self.selected_logits.extend(other.selected_logits)
        self.token_is_text.extend(other.token_is_text)

    def finalize(self) -> dict:
        if not self.topk_ids:
            zero = torch.zeros(self.num_experts)
            return {
                "text_salience": zero.clone(), "nontext_salience": zero.clone(),
                "text_frequency": zero.clone(), "nontext_frequency": zero.clone(),
                "tau_text": 0.0, "tau_nontext": 0.0,
                "text_tokens": 0, "nontext_tokens": 0,
            }
        ids = torch.cat(self.topk_ids, dim=0).long()
        weights = torch.cat(self.topk_weights, dim=0)
        logits = torch.cat(self.selected_logits, dim=0)
        is_text = torch.cat(self.token_is_text, dim=0)
        # Pangu multiplies normalized routing weights by routed_scaling_factor.
        # SGFR confidence should remain a probability-like quantity.
        probs = weights / max(self.routed_scaling_factor, 1e-12)

        result = {}
        for name, modality in (("text", is_text), ("nontext", ~is_text)):
            salience = torch.zeros(self.num_experts)
            frequency = torch.zeros(self.num_experts)
            count = int(modality.sum().item())
            if count:
                p = probs[modality]
                eid = ids[modality]
                g = logits[modality]
                tau = float(p.amax(dim=1).mean().item())
                confident = p >= tau
                # Apply the filter after rectification. Applying sigmoid to a
                # zeroed logit would incorrectly give every rejected route 0.5.
                contribution = torch.maximum(g, torch.sigmoid(g)) * confident
                salience.scatter_add_(0, eid.reshape(-1), contribution.reshape(-1))
                frequency.scatter_add_(
                    0, eid.reshape(-1), torch.ones_like(p).reshape(-1))
            else:
                tau = 0.0
            result[f"{name}_salience"] = salience
            result[f"{name}_frequency"] = frequency
            result[f"tau_{name}"] = tau
            result[f"{name}_tokens"] = count
        return result
