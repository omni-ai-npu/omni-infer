# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
MDMixQ-guided JointFix with a uniform W8A8 deployment format.

MDMixQ contributes modality-decoupled calibration, SGFR route salience and a
text-first calibration budget.  It does not assign mixed bit widths here.
"""
from __future__ import annotations

from jointfix.methods.jointfix import JointFixMethod, JointSearchConfig
from jointfix.registry import register_method


@register_method("jointfix-mdmixq")
class JointFixMDMixQMethod(JointFixMethod):
    name = "jointfix-mdmixq"

    def __init__(self, config: JointSearchConfig | None = None):
        config = config or JointSearchConfig()
        config.mdmixq_enabled = True
        super().__init__(config)

    def configure(self, args) -> None:
        super().configure(args)
        for field in (
            "mdmixq_text_sample_ratio",
            "mdmixq_text_expert_fraction",
            "mdmixq_nontext_topk",
            "mdmixq_candidate_multiplier",
        ):
            if hasattr(args, field):
                setattr(self.config, field, getattr(args, field))
        if not 0.0 <= self.config.mdmixq_text_sample_ratio <= 1.0:
            raise ValueError("--mdmixq-text-sample-ratio must be in [0,1]")
        if not 0.0 < self.config.mdmixq_text_expert_fraction <= 1.0:
            raise ValueError("--mdmixq-text-expert-fraction must be in (0,1]")

    def add_cli_args(self, parser) -> None:
        super().add_cli_args(parser)
        g = parser.add_argument_group(
            "jointfix-mdmixq: modality-decoupled uniform W8A8 calibration")
        g.add_argument("--mdmixq-text-sample-ratio", type=float, default=0.8)
        g.add_argument("--mdmixq-text-expert-fraction", type=float, default=0.25)
        g.add_argument("--mdmixq-nontext-topk", type=int, default=8)
        g.add_argument("--mdmixq-candidate-multiplier", type=int, default=4)
