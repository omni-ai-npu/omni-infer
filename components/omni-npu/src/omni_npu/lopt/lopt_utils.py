# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from functools import reduce
from typing import Iterable, List, Sequence, Union
import numpy as np
import torch


def chunks(sentence: Union[str, Sequence[str]], chunk_size: int = 40960, overlap_length: int = 512):
    if isinstance(sentence, str):
        while len(sentence) - chunk_size > 100:
            yield sentence[: overlap_length + chunk_size]
            sentence = sentence[chunk_size:]
        yield sentence
    elif isinstance(sentence, Sequence):
        if not sentence:
            return
        if not isinstance(sentence[0], str):
            raise ValueError(f"Unsupported type {type(sentence[0])} for chunks, expected str") 
        remaining = list(sentence)
        while any(remaining):
            yield [s[: overlap_length + chunk_size] if len(s[chunk_size:]) > 100 else s for s in remaining]
            remaining = [s[chunk_size:] if len(s[chunk_size:]) > 100 else "" for s in remaining]
    else:
        raise ValueError(f"Unsupported type {type(sentence)} for chunks")


def pairs(chunk_list: List[List[int]]) -> Iterable[List[List[int]]]:
    for i in range(0, len(chunk_list) - 1):
        yield (chunk_list[i], chunk_list[i + 1])


def flatten(item: Union[torch.Tensor, np.ndarray, List]):
    if isinstance(item, (torch.Tensor, np.ndarray)):
        return item.flatten()
    elif isinstance(item, List):
        while all(isinstance(i, Sequence) for i in item):
            item = [i for sublist in item for i in sublist]
        if any(isinstance(i, Sequence) for i in item):
            raise ValueError("flatten failed, still contains nested sequences")
        return item
    else:
        raise ValueError(f"Unsupported type {type(item)} for flatten")
