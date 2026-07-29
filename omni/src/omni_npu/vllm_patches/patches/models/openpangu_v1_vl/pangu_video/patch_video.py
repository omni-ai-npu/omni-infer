# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import numpy.typing as npt
from typing import Any, Optional
import tempfile
import os
import warnings
import cv2
from PIL import Image
import itertools
from vllm.multimodal.video import VIDEO_LOADER_REGISTRY, OpenCVVideoBackend


def get_extracted_frame_indices(total_frames: int, original_fps: float, target_fps: float):
    """
    align with train extract image frames from video 
    """
    if total_frames <= 0:
        return []

    frame_step = original_fps / target_fps
    frame_indices = []
    idx = 0
    count_float = 0.0

    for raw_frame_idx in range(total_frames):
        count = round(count_float)
        idx += 1
        if idx <= count:
            continue
        frame_indices.append(raw_frame_idx)
        count_float += frame_step
    return frame_indices


class NPUOpenCVDynamicVideoBackend(OpenCVVideoBackend):
    @staticmethod
    def decode_single_frame(frame_idx: int, video_path: str, total_frames_num: int) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                next_idx = frame_idx + 1
                while next_idx < total_frames_num:
                    ret_next, next_frame = cap.read()
                    if ret_next:
                        return cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
                    next_idx += 1
                return None
        finally:
            cap.release()

    @classmethod
    def _resolve_sampled_frames(
        cls,
        frame_sample_strategy: str,
        total_frames_num: int,
        original_fps: float,
        total_duration: float,
        num_frames: int,
        sample_fps: int,
        kwargs: dict[str, Any],
    ) -> tuple[list[int], Any, float]:
        strategy_handlers = {
            "uniform": cls._sample_uniform_frames,
            "fps-uniform": cls._sample_uniform_frames,
            "offline": cls._sample_offline_frames,
        }
        sample_func = strategy_handlers.get(frame_sample_strategy)
        if sample_func is None:
            raise NotImplementedError

        return sample_func(
            total_frames_num=total_frames_num,
            original_fps=original_fps,
            total_duration=total_duration,
            num_frames=num_frames,
            sample_fps=sample_fps,
            kwargs=kwargs,
        )

    @classmethod
    def _sample_uniform_frames(
        cls,
        total_frames_num: int,
        original_fps: float,
        total_duration: float,
        num_frames: int,
        sample_fps: int,
        kwargs: dict[str, Any],
    ) -> tuple[list[int], list[float], float]:
        clip_delta = kwargs.pop("clip_delta", 0) 
        (
            abs_sample_frame_timestamps_list, 
            per_clip_sample_frame_timestamps_list,
        ) = cls.smart_sample_multi_clips(
            num_frame=num_frames, 
            clip_duration_list=[total_duration],
            sample_fps=sample_fps,
            clip_delta_list=[clip_delta]) 
            # per_clip_sample_frame_timestamps_list 转换为 per_clip_frame_indices_list
        frames_indices = []
        for frame_time in per_clip_sample_frame_timestamps_list[0]:
            frame_idx = min(total_frames_num - 1, round(frame_time * original_fps))
            frames_indices.append(frame_idx)

        sample_frame_timestamps = abs_sample_frame_timestamps_list[0]
        return frames_indices, sample_frame_timestamps, total_duration

    @staticmethod
    def _sample_offline_frames(
        total_frames_num: int,
        original_fps: float,
        total_duration: float,
        num_frames: int,
        sample_fps: int,
        kwargs: dict[str, Any],
    ) -> tuple[list[int], npt.NDArray, float]:
        if sample_fps > 0:
            extracted_frame_indices = get_extracted_frame_indices(
                total_frames=total_frames_num,
                original_fps=original_fps,
                target_fps=sample_fps,
            )
            extracted_total_frames = len(extracted_frame_indices)
            # align with train
            total_duration = (extracted_total_frames - 1) / sample_fps
            # Num_frames is the maximum number of frames to sample. 
            # If fewer frames are sampled at this sample_fps, the update duration will be longer.
            if num_frames >= int(total_duration * sample_fps) + 1:
                num_frames = int(total_duration * sample_fps) + 1
                # Under the new maximum frame rate, the video duration of the rightmost frame,
                # cannot be calculated for frame 0.
                total_duration = min(total_duration, (num_frames - 1) / sample_fps) 
                
            sample_frame_timestamps = np.linspace(0, total_duration, num_frames, dtype=float)
            sampled_seq_indices = [
                min(extracted_total_frames - 1, round(t * sample_fps))
                for t in sample_frame_timestamps
            ]
            # align with train:get the train offline extract frame index
            frames_indices = [extracted_frame_indices[idx] for idx in sampled_seq_indices]
            return frames_indices, sample_frame_timestamps, total_duration

        elif sample_fps == -1:
            # train sample_fps > 0,so keep the old logic 
            sample_frame_timestamps = np.linspace(0, total_duration, num_frames, dtype=float)
            frames_indices = [
                min(total_frames_num - 1, round(t * original_fps))
                for t in sample_frame_timestamps
            ]
            return frames_indices, sample_frame_timestamps, total_duration

        raise ValueError(f"requires dataset fps is -1 or greater than 0 but got {sample_fps}")

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = 32,
        sample_fps: int = 1,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        temp_path = None
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(data)
            temp_path = f.name
        cap = cv2.VideoCapture(temp_path) 
        if not cap.isOpened():
            raise ValueError("Could not open video stream")

        total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of video frames
        original_fps = float(cap.get(cv2.CAP_PROP_FPS)) # Video fps 
        # The timestamp of the rightmost frame, cannot be used to calculate frame 0.
        total_duration = (total_frames_num - 1) / original_fps

        # `sample_fps` is the FPS parameter passed in for sampling,
        # -1 indicates that sampling can be performed directly without FPS limitation.
        frame_sample_strategy = kwargs.pop("frame_sample_strategy", "fps-uniform") 
        if frame_sample_strategy == 'uniform':
            sample_fps = -1 # 表示不使用fps作为采样限制
        elif isinstance(sample_fps, list):
            import random
            sample_fps = random.choice(sample_fps)

        (
            frames_indices, 
            sample_frame_timestamps, 
            total_duration,
        ) = cls._resolve_sampled_frames(
            frame_sample_strategy=frame_sample_strategy,
            total_frames_num=total_frames_num,
            original_fps=original_fps,
            total_duration=total_duration,
            num_frames=num_frames,
            sample_fps=sample_fps,
            kwargs=kwargs,
        )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.empty((len(frames_indices), height, width, 3), dtype=np.uint8)
        valid_count = 0
        task = [(frame_idx, temp_path, total_frames_num) for frame_idx in frames_indices]
        decode_frame_thread_count = kwargs.pop("decode_frame_thread_count", 4)
        with ThreadPoolExecutor(max_workers=decode_frame_thread_count) as executor:
            results = list(executor.map(lambda args: cls.decode_single_frame(*args), task))
            for frame_data in results:
                if frame_data is not None:
                    frames[valid_count] = frame_data
                    valid_count += 1

        if valid_count != len(frames_indices):
            warnings.warn(
                f"Expected reading {len(frames_indices)} frames, "
                f"but only loaded {valid_count} frames from video.",
                UserWarning,
                stacklevel=2
            )
        if temp_path:
            os.remove(temp_path)

        # Use transformers transformers.video_utils.VideoMetadata format.
        metadata = {
            "total_num_frames": total_frames_num,
            "fps": original_fps,
            "duration": total_duration,
            "video_backend": "opencv_dynamic",
            "frames_indices": frames_indices,
            "do_sample_frames": False,
            "sample_frame_timestamps": sample_frame_timestamps
        }
        return frames, metadata

    @staticmethod
    def smart_sample_multi_clips(num_frame, 
                                 clip_duration_list, 
                                 sample_fps,
                                 clip_delta_list, 
                                 ):
        """
            对同一视频的多段clip做统一采样
            输入：
                max_num_frame：假设总帧数最大限制在 max_num_frame 以内
                clip_duration_list：记录每段video clip的持续时长
                sample_fps: 
                clip_delta_list: 
            输出：
                abs_sample_frame_timestamps_list：采样帧的绝对时间戳，按clip隔开
                per_clip_sample_frame_timestamps_list: 采样帧的相对每个clip的时间戳，按clip隔开
        """
        total_duration = sum(clip_duration_list) + sum(clip_delta_list) - clip_delta_list[0] # 不能算首帧偏移
        if sample_fps > 0:
            if num_frame >= int(total_duration * sample_fps) + 1:
                num_frame = int(total_duration * sample_fps) + 1 # 新的最大帧数，需要计算第0帧
                total_duration = min(total_duration, (num_frame - 1) / sample_fps) # 在新最大帧数下，最右帧的视频时长(未计0帧偏移量)
        elif sample_fps != -1: 
            raise ValueError(f"requires dataset fps is -1 or greater than 0 but got {sample_fps}")
        # 按时间戳均匀采样
        sample_frame_timestamps = clip_delta_list[0] + np.linspace(0, total_duration, num_frame, dtype=float)

        # 计算每个片段相对全局的偏移时间
        clip_accum_durations = [0] + list(itertools.accumulate(clip_duration_list))
        clip_delta_list = clip_delta_list + [0] # 最后一个片段没有尾部偏移
        clip_accum_deltas = list(itertools.accumulate(clip_delta_list)) # 计算每一个片段尾部的偏移
        clip_timestamps = [duration + delta for duration, delta in zip(clip_accum_durations, clip_accum_deltas)]

        # 把绝对时间戳映射到每个子片段的相对时间戳
        cursor = 0
        abs_sample_frame_timestamps_list = []
        per_clip_sample_frame_timestamps_list = []
        for i, clip_start in enumerate(clip_timestamps[:-1]):
            clip_end = clip_timestamps[i + 1] - clip_delta_list[i + 1] # 当前片段不能算上尾部的偏移
            abs_sample_frame_timestamps = []
            per_clip_sample_frame_timestamps = []
            while cursor < num_frame and sample_frame_timestamps[cursor] <= clip_end:
                frame_timestamp = sample_frame_timestamps[cursor]
                if frame_timestamp - clip_start >= 0: # 如果恰好采在clip_delta上，则视为帧损坏，跳过
                    abs_sample_frame_timestamps.append(frame_timestamp)
                    per_clip_sample_frame_timestamps.append(frame_timestamp - clip_start)
                cursor += 1
            abs_sample_frame_timestamps_list.append(abs_sample_frame_timestamps)
            per_clip_sample_frame_timestamps_list.append(per_clip_sample_frame_timestamps)
        return abs_sample_frame_timestamps_list, per_clip_sample_frame_timestamps_list

# registerbackend
VIDEO_LOADER_REGISTRY.register("npu_opencv_dynamic")(NPUOpenCVDynamicVideoBackend)
