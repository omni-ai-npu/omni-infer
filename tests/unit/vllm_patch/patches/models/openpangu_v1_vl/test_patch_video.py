# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Real NumPy sampling; small capture doubles isolate codec-independent logic."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")


@pytest.fixture
def video(patch_env):
    cv2 = patch_env.stub(
        "cv2", VideoCapture=Mock(), CAP_PROP_POS_FRAMES=1, CAP_PROP_FRAME_COUNT=7,
        CAP_PROP_FPS=5, CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
        COLOR_BGR2RGB=4,
        cvtColor=Mock(side_effect=lambda frame, code: frame[..., ::-1].copy()),
    )
    registry = {}
    patch_env.stub(
        "vllm.multimodal.video", VideoLoader=type("VideoLoader", (), {}),
        OpenCVVideoBackendMixin=type("OpenCVVideoBackendMixin", (), {}),
        VIDEO_LOADER_REGISTRY=SimpleNamespace(
            register=lambda name: lambda cls: registry.setdefault(name, cls)),
    )
    mod = patch_env.load("patch_video")
    assert registry.get("npu_opencv_dynamic") is mod.NPUOpenCVDynamicVideoBackend
    return SimpleNamespace(mod=mod, backend=mod.NPUOpenCVDynamicVideoBackend, cv2=cv2)


@pytest.mark.parametrize("count,fps,target,expected", [
    (0, 30, 2, []), (1, 30, 2, [0]), (5, 4, 2, [0, 2, 4]),
    (8, 5, 2, [0, 2, 5]), (4, 2, 4, [0, 1, 2, 3]),
])
def test_training_frame_extraction_rounding(video, count, fps, target, expected):
    assert video.mod.get_extracted_frame_indices(count, fps, target) == expected


@pytest.mark.parametrize("strategy,fps,limit,indices,times,duration", [
    ("uniform", -1, 4, [0, 3, 5, 8], [0, 2 / 3, 4 / 3, 2], 2),
    ("fps-uniform", 2, 99, [0, 2, 4, 6, 8], [0, .5, 1, 1.5, 2], 2),
    ("fps-uniform", 2, 3, [0, 4, 8], [0, 1, 2], 2),
    ("offline", 2, 99, [0, 2, 4, 6, 8], [0, .5, 1, 1.5, 2], 2),
    ("offline", 2, 3, [0, 4, 8], [0, 1, 2], 2),
    ("offline", -1, 4, [0, 3, 5, 8], [0, 2 / 3, 4 / 3, 2], 2),
])
def test_sampling_strategies_use_real_timestamps(video, strategy, fps, limit, indices, times, duration):
    actual, timestamps, actual_duration = video.backend._resolve_sampled_frames(
        strategy, 9, 4, 2, limit, fps, {})
    assert actual == indices
    np.testing.assert_allclose(timestamps, times, rtol=0, atol=1e-12)
    assert actual_duration == duration


@pytest.mark.parametrize("strategy", ["fps-uniform", "offline"])
def test_single_frame_video_caps_sampling_at_one(video, strategy):
    indices, times, duration = video.backend._resolve_sampled_frames(strategy, 1, 30, 0, 128, 2, {})
    assert indices == [0]
    np.testing.assert_array_equal(times, [0])
    assert duration == 0


def test_uniform_clip_offset_changes_metadata_not_frame_indices(video):
    indices, times, duration = video.backend._resolve_sampled_frames(
        "uniform", 9, 4, 2, 3, -1, {"clip_delta": 10})
    assert indices == [0, 4, 8]
    assert times == [10, 11, 12]
    assert duration == 2


def test_offline_duration_uses_extracted_frames(video):
    indices, times, duration = video.backend._resolve_sampled_frames(
        "offline", 8, 5, 1.4, 99, 2, {})
    assert indices == [0, 2, 5]
    np.testing.assert_array_equal(times, [0, .5, 1])
    assert duration == 1


@pytest.mark.parametrize("fps", [-1, 1])
def test_multi_clip_sampling_skips_gaps_and_preserves_absolute_offsets(video, fps):
    absolute, relative = video.backend.smart_sample_multi_clips(8, [2, 3], fps, [10, 2])
    assert absolute == [[10, 11, 12], [14, 15, 16, 17]]
    assert relative == [[0, 1, 2], [0, 1, 2, 3]]


def test_adjacent_clip_boundary_is_not_duplicated(video):
    absolute, relative = video.backend.smart_sample_multi_clips(5, [2, 2], -1, [0, 0])
    assert absolute == [[0, 1, 2], [3, 4]]
    assert relative == [[0, 1, 2], [1, 2]]


@pytest.mark.parametrize("strategy", ["fps-uniform", "offline"])
@pytest.mark.parametrize("fps", [0, -2])
def test_invalid_fps_keeps_existing_error(video, strategy, fps):
    with pytest.raises(ValueError, match="dataset fps"):
        video.backend._resolve_sampled_frames(strategy, 9, 4, 2, 3, fps, {})


def test_unknown_strategy_keeps_existing_error(video):
    with pytest.raises(NotImplementedError):
        video.backend._resolve_sampled_frames("unknown", 9, 4, 2, 3, 2, {})


@pytest.mark.parametrize("failed_reads", [0, 1, 2])
def test_frame_read_seeks_once_falls_forward_and_converts_color(video, failed_reads):
    frame = np.array([[[10, 20, 30]]], dtype=np.uint8)
    cap = Mock()
    cap.read.side_effect = [(False, None)] * failed_reads + [(True, frame)]
    video.cv2.VideoCapture.return_value = cap
    result = video.backend.decode_single_frame(2, "video.mp4", 5)
    np.testing.assert_array_equal(result, [[[30, 20, 10]]])
    cap.set.assert_called_once_with(video.cv2.CAP_PROP_POS_FRAMES, 2)
    assert cap.read.call_count == failed_reads + 1
    cap.release.assert_called_once()


def test_unreadable_tail_returns_none_and_releases_capture(video):
    cap = Mock()
    cap.read.return_value = (False, None)
    video.cv2.VideoCapture.return_value = cap
    assert video.backend.decode_single_frame(3, "video.mp4", 5) is None
    assert cap.read.call_count == 2
    cap.release.assert_called_once()


def test_decode_exception_still_releases_capture(video):
    cap = Mock()
    error = RuntimeError("codec failed")
    cap.read.side_effect = error
    video.cv2.VideoCapture.return_value = cap
    with pytest.raises(RuntimeError) as caught:
        video.backend.decode_single_frame(0, "video.mp4", 1)
    assert caught.value is error
    cap.release.assert_called_once()


@pytest.mark.parametrize("strategy,fps,expected", [
    ("uniform", 1, [0, 4, 8]),
    ("fps-uniform", [2], [0, 4, 8]),
    ("offline", 2, [0, 4, 8]),
])
def test_load_bytes_joins_actual_sampling_and_frame_decode(video, strategy, fps, expected):
    captures = []
    metadata = {7: 9, 5: 4, 3: 1, 4: 1}

    def open_capture(path):
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.side_effect = metadata.__getitem__
        cap.position = 0

        def seek(prop, frame):
            cap.position = frame

        def read():
            return True, np.array([[[cap.position, 20, 30]]], dtype=np.uint8)

        cap.set.side_effect = seek
        cap.read.side_effect = read
        captures.append(cap)
        return cap

    video.cv2.VideoCapture.side_effect = open_capture
    frames, metadata = video.backend.load_bytes(
        b"capture double", num_frames=3, sample_fps=fps,
        frame_sample_strategy=strategy, decode_frame_thread_count=2)
    np.testing.assert_array_equal(frames[:, 0, 0], [[30, 20, i] for i in expected])
    assert frames.dtype == np.uint8
    assert metadata["frames_indices"] == expected
    np.testing.assert_array_equal(metadata["sample_frame_timestamps"], [0, 1, 2])
    assert metadata["total_num_frames"] == 9 and metadata["duration"] == 2
    assert metadata["fps"] == 4 and metadata["do_sample_frames"] is False
    for cap in captures:
        cap.release.assert_called_once()
