# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from omni_npu.lopt import (
    LOPT_AVAILABLE,
    LoptParallelTokenizer,
    maybe_get_lopt_tokenizer,
)


class TestLoptInitExports:
    def test_lopt_available_is_bool(self):
        assert isinstance(LOPT_AVAILABLE, bool)

    def test_lopt_parallel_tokenizer_is_class(self):
        assert isinstance(LoptParallelTokenizer, type)

    def test_maybe_get_lopt_tokenizer_is_callable(self):
        assert callable(maybe_get_lopt_tokenizer)

    def test_all_exports(self):
        from omni_npu import lopt

        assert hasattr(lopt, "LOPT_AVAILABLE")
        assert hasattr(lopt, "LoptParallelTokenizer")
        assert hasattr(lopt, "maybe_get_lopt_tokenizer")
