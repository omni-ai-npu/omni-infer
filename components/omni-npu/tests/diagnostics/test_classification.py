# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

# SPDX-License-Identifier: MIT
"""Unit tests for omni_npu.diagnostics.classification (stdlib-only, no NPU)."""

import pytest

from omni_npu.diagnostics import classification as cls


class TestDotPathCodec:
    @pytest.mark.parametrize("segments", [
        ["vllm", "parallel", "tensor_parallel_size"],
        ["model", "omni", "parall_config", "layer_parallel_config",
         "self_attn.o_proj", "tp_size_or_ranks"],          # real dotted key
        ["a", 0, "b"],                                      # array index
        ["a", "0"],                                         # numeric STRING key
        ["a", 'with"quote', "b"],
        ["a", "with]bracket", "b"],
        ["a", "back\\slash"],
        ["a", "spa ced", "x"],
        ["a", "k=v"],
    ])
    def test_roundtrip(self, segments):
        assert cls.decode_path(cls.encode_segments(segments)) == segments

    def test_quoted_vs_index_unambiguous(self):
        s_idx = cls.encode_segments(["a", 0])
        s_key = cls.encode_segments(["a", "0"])
        assert s_idx != s_key
        assert cls.decode_path(s_idx) == ["a", 0]
        assert cls.decode_path(s_key) == ["a", "0"]

    def test_dotted_key_encoding_shape(self):
        path = cls.encode_segments(["layer_parallel_config", "self_attn.o_proj"])
        assert path == 'layer_parallel_config["self_attn.o_proj"]'

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            cls.decode_path("a.[")
        with pytest.raises(ValueError):
            cls.decode_path("a.6bad")

    def test_malformed_int_index_has_path_context(self):
        # review (omni_ci): int-index parse errors must carry the offending
        # path, and a non-numeric index must raise ValueError (consistent type).
        with pytest.raises(ValueError) as ei:
            cls.decode_path("a[0")        # missing closing bracket
        assert "a[0" in str(ei.value)
        with pytest.raises(ValueError) as ei:
            cls.decode_path("a[xy]")      # non-numeric array index
        assert "a[xy]" in str(ei.value)

    def test_malformed_json_segment_has_path_context(self):
        # review (omni_ci): a broken JSON string segment must be wrapped with
        # the path, not leak a bare json.JSONDecodeError without context.
        with pytest.raises(ValueError) as ei:
            cls.decode_path('a["oops')    # unterminated JSON string segment
        assert "oops" in str(ei.value)

    def test_stray_dot_rejected(self):
        # review (omni_ci, line 245): the '.' separator is valid only BETWEEN
        # segments and before a bare identifier; leading / trailing / doubled
        # dots (and '.'-before-bracket) are malformed - the encoder never emits
        # them, so rejecting them keeps the documented "raises on malformed"
        # contract honest without affecting any encode->decode round-trip.
        for bad in ("a..b", ".a", "a.", "a.[0]"):
            with pytest.raises(ValueError):
                cls.decode_path(bad)

    def test_unterminated_string_segment_rejected(self):
        # raw_decode SUCCEEDS but the closing ']' is missing or wrong - distinct
        # from test_malformed_json_segment_has_path_context (which fails inside
        # raw_decode on a broken JSON string).
        for bad in ('a["abc"', 'a["abc"x'):
            with pytest.raises(ValueError) as ei:
                cls.decode_path(bad)
            assert "unterminated string segment" in str(ei.value)

    def test_leading_non_identifier_rejected(self):
        # a path starting with neither '.', '[' nor an identifier char is
        # malformed and must fail loudly with the offset in the message.
        for bad in ("9abc", "#foo", "-x"):
            with pytest.raises(ValueError):
                cls.decode_path(bad)


class TestClassification:
    def test_identity_and_volatile(self):
        assert cls.classify_key("meta.rank") == cls.CLASS_IDENTITY
        assert cls.classify_key("env.ASCEND_RT_VISIBLE_DEVICES") == cls.CLASS_IDENTITY
        assert cls.classify_key("vllm.kv_transfer.kv_port") == cls.CLASS_VOLATILE

    def test_real_config_stays_shared(self):
        # codex round-2 regression: these were swallowed by broad regexes once
        for k in ("env.ENABLE_OMNI_CACHE", "env.KV_CACHE_MODE",
                  "env.CUSTOM_MODEL_CONFIG_PATH", "env.HCCL_IF_BASE_PORT",
                  "env.RANK_TABLE_FILE"):
            assert cls.classify_key(k) == cls.CLASS_SHARED, k

    def test_unknown_defaults_to_shared(self):
        assert cls.classify_key("vllm.scheduler.some_future_field") == cls.CLASS_SHARED
        assert cls.classify_key("env.SOME_RANDOM_VAR") == cls.CLASS_SHARED


class TestSensitiveMask:
    @pytest.mark.parametrize("name", [
        "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY",     # codex round-5 leak pair
        "VLLM_API_KEY", "OPENAI_API_KEY",
        "MY_SERVICE_TOKEN", "FOO_ACCESS_KEY", "BAR_SECRET_THING", "DB_PASSWORD",
    ])
    def test_credentials_masked(self, name):
        assert cls.is_sensitive(f"env.{name}") is True

    @pytest.mark.parametrize("name", [
        "ENABLE_OMNI_CACHE", "KV_CACHE_MODE", "CUSTOM_MODEL_CONFIG_PATH",
        "HCCL_IF_BASE_PORT", "TOKENIZERS_PARALLELISM",
    ])
    def test_real_config_not_masked(self, name):
        assert cls.is_sensitive(f"env.{name}") is False

    def test_mask_limited_to_env_namespace(self):
        # codex round-4 regression: substring matching once masked these
        assert cls.is_sensitive("model.hf.num_key_value_heads") is False
        assert cls.is_sensitive("vllm.scheduler.max_num_batched_tokens") is False
        assert cls.is_sensitive("vllm.model.tokenizer_mode") is False

    def test_explicit_sensitive_path_masked(self, monkeypatch):
        # SENSITIVE_PATHS is empty by default; when an operator registers a
        # non-env path it must mask regardless of namespace (the env regex would
        # never catch a vllm.* path).
        monkeypatch.setattr(cls, "SENSITIVE_PATHS",
                            frozenset({"vllm.secret_field"}))
        assert cls.is_sensitive("vllm.secret_field") is True
        assert cls.is_sensitive("vllm.other_field") is False


class TestEnvNameOf:
    def test_non_env_path_returns_none(self):
        assert cls.env_name_of("vllm.parallel.rank") is None

    def test_simple_and_bracketed_env_names(self):
        assert cls.env_name_of("env.RANK_ID") == "RANK_ID"
        # k8s-style hyphen twin is encoded as a bracketed JSON-string segment
        assert cls.env_name_of('env["NPU-VISIBLE-DEVICES"]') == "NPU-VISIBLE-DEVICES"

    def test_malformed_env_path_swallowed_to_none(self):
        # starts with 'env' but decode_path raises -> ValueError swallowed, None
        assert cls.env_name_of('env["oops') is None

    def test_non_two_segment_env_returns_none(self):
        assert cls.env_name_of("env.a.b") is None   # 3 segments
        assert cls.env_name_of("env[0]") is None     # 2nd segment is an int index


class TestEnvWhitelist:
    def test_prefixes_and_exact(self):
        assert cls.env_is_collected("ASCEND_RT_VISIBLE_DEVICES")
        assert cls.env_is_collected("HCCL_BUFFSIZE")
        assert cls.env_is_collected("VLLM_PLUGINS")
        assert cls.env_is_collected("ROLE")
        assert not cls.env_is_collected("HOME")
        assert not cls.env_is_collected("PATH")
