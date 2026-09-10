# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import torch

from jointfix.core.modality import MoERouteStats


def test_route_stats_separate_text_and_nontext_and_mask_after_rectifier():
    stats = MoERouteStats(3, routed_scaling_factor=2.0)
    stats.update(
        topk_ids=torch.tensor([[0, 1], [1, 2]]),
        topk_weights=torch.tensor([[1.6, 0.4], [1.2, 0.8]]),
        selected_logits=torch.tensor([[2.0, -4.0], [1.0, -3.0]]),
        token_is_text=torch.tensor([True, False]),
    )
    out = stats.finalize()
    assert out["text_tokens"] == 1 and out["nontext_tokens"] == 1
    assert out["text_salience"][0] > 0
    assert out["text_salience"][1] == 0  # rejected route contributes no sigmoid(0)=0.5
    assert out["nontext_salience"][1] > 0
    assert out["nontext_salience"][2] == 0
