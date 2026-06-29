# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
import multiprocessing as mp
from functools import reduce
from typing import Optional

import numpy as np
from transformers import AutoTokenizer, BatchEncoding
import weakref

try:
    import Cpp_match_merge
    from .lopt_utils import chunks, flatten, pairs

    LOPT_AVAILABLE = True
except ImportError:
    LOPT_AVAILABLE = False

logger = logging.getLogger(__name__)
_worker_tokenizer = None


class LoptParallelTokenizer:
    """LOPT parallel tokenizer wrapper for vLLM integration.

    This class provides parallel tokenization for long texts by:
    1. Splitting text into overlapping chunks
    2. Processing chunks in parallel using multiple processes
    3. Matching and merging overlapping regions using C++ extension
    4. Returning results compatible with HF tokenizer format
    """

    def __init__(
        self,
        model_path: str,
        pool_size: int = 8,
        chunk_size: int = 2048,
        overlap_ratio: float = 0.125,
    ):
        self.model_path = model_path
        self.pool_size = pool_size
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size * overlap_ratio)
        self.threshold = 2

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )

        self.pool = mp.Pool(
            pool_size,
            initializer=self._init_worker,
            initargs=(model_path,),
        )
        self._finalizer = weakref.finalize(self, self.close)
        
    def close(self):
        """close the multiprocessing pool if it exists"""
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()       # Terminate the worker processes immediately
            self.pool.join()
            self.pool = None

    @staticmethod
    def _init_worker(model_path: str):
        global _worker_tokenizer
        _worker_tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )

    @staticmethod
    def _tokenize_chunk(text: str):
        global _worker_tokenizer
        return _worker_tokenizer(
            text,
            return_tensors="np",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

    def _cpp_match_wrapper(
        self, tokens_a: np.ndarray, tokens_b: np.ndarray, chunk_size: int
    ) -> tuple[int, int]:
        a = np.ascontiguousarray(tokens_a, dtype=np.int64)
        b = np.ascontiguousarray(tokens_b, dtype=np.int64)
        res = Cpp_match_merge.match(a, b, chunk_size, self.threshold)
        if res[0] < 0:
            raise RuntimeError(f"C++ match returned error: {res}")
        return res

    def __call__(
        self, text: str, add_special_tokens: bool = False
    ) -> "BatchEncoding":
        if len(text) < self.chunk_size * 2:
            result = self.tokenizer(text, add_special_tokens=add_special_tokens)
            return result
        else:
            result = self._parallel_encode(text, add_special_tokens=add_special_tokens)
            return result

    def encode(
        self,
        text: str,
        add_special_tokens: bool,
    ) -> list[int]:
        return self.__call__(text, add_special_tokens).input_ids

    def _parallel_encode(self, text: str, add_special_tokens: bool) -> "BatchEncoding":
        text_chunks = list(chunks(text, self.chunk_size, self.overlap))

        shards = self.pool.map(self._tokenize_chunk, text_chunks)

        tokens_shards = [
            flatten(shard["offset_mapping"])[::2] for shard in shards
        ]

        try:
            matches = [
                self._cpp_match_wrapper(_[0], _[1], self.chunk_size)
                for _ in pairs(tokens_shards)
            ]
            matches = (
                [len(tokens_shards[0])]
                + list(reduce(lambda x, y: x + y, matches))
                + [0]
            )
        except RuntimeError:
            logger.warning("Fall back to standard tokenizer on match failure")
            return self.tokenizer(text, return_tensors="np", add_special_tokens=add_special_tokens)

        merged = BatchEncoding({})

        for key in shards[0].keys():
            if key not in ["offset_mapping", "attention_mask"]:
                merged[key] = Cpp_match_merge.merge(
                    [shard[key] for shard in shards], matches
                )[np.newaxis, :].tolist()[0]
            if key == "input_ids" and add_special_tokens:
                merged[key] = self.tokenizer.build_inputs_with_special_tokens(
                    merged[key]
                )

        return merged


def maybe_get_lopt_tokenizer(
    model_path: str,
    enable_lopt: bool = False,
    lopt_pool_size: int = 8,
    lopt_chunk_size: int = 2048,
) -> Optional[LoptParallelTokenizer]:
    if not enable_lopt:
        return None

    if not LOPT_AVAILABLE:
        logger.warning(
            "LOPT was requested but Cpp_match_merge module is not available. "
            "Falling back to standard tokenization."
        )
        return None

    try:
        return LoptParallelTokenizer(
            model_path=model_path,
            pool_size=lopt_pool_size,
            chunk_size=lopt_chunk_size,
        )
    except Exception as e:
        logger.warning(f"Failed to initialize LOPT tokenizer: {e}")
        return None
