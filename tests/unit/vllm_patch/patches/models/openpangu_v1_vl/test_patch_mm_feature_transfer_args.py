# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import argparse
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("config", [None, "", '{"connectors": {}}', "not-json"])
def test_registration_is_opt_in_and_errors_are_not_swallowed(patch_env, config):
    class EngineArgs:
        @staticmethod
        def add_cli_args(parser):
            parser.add_argument("--model")
            return parser

        @classmethod
        def from_cli_args(cls, args):
            return cls(model=args.model)

        def __init__(self, model):
            self.model = model

    patch_env.stub("vllm", EngineArgs=EngineArgs)
    patch_env.stub("vllm.utils.argparse_utils", FlexibleArgumentParser=argparse.ArgumentParser)
    register = Mock()
    patch_env.stub("omni_npu.connector.mm_feature_transfer", register=register)
    mod = patch_env.load("patch_mm_feature_transfer_args")
    mod.MMFeatureTransferArgsPatch.apply()
    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    cli = ["--model", "test-model"]
    if config is not None:
        cli += ["--mm-feature-transfer-config", config]
    args = parser.parse_args(cli)
    if config == "not-json":
        error = ValueError("connector configuration is invalid")
        register.side_effect = error
        with pytest.raises(ValueError) as caught:
            EngineArgs.from_cli_args(args)
        assert caught.value is error
    else:
        instance = EngineArgs.from_cli_args(args)
        assert instance.model == "test-model"
        assert vars(instance) == {"model": "test-model"}
    if config:
        register.assert_called_once_with(config)
    else:
        register.assert_not_called()
