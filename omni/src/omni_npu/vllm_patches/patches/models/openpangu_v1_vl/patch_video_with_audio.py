# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import torch
import os
from typing import Tuple, Dict, Union
from io import BytesIO
from urllib.parse import urlparse
from pathlib import Path
import base64
import numpy.typing as npt

from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.utils.import_utils import PlaceholderModule
from vllm.entrypoints.chat_utils import AsyncMultiModalContentParser
try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

DEFAULT_AUDIO_SAMPLE_RATE = 16000


def _run_ffmpeg_extract(source_path: str, sampling_rate: int) -> bytes:
    import ffmpeg


    try:
        audio_output, _ = (
            ffmpeg
            .input(source_path)
            .output("pipe:", format="wav", acodec="pcm_s16le", ar=str(sampling_rate), vn=None)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return audio_output
    except ffmpeg.Error as e:
        stderr_msg = (
            e.stderr.decode('utf-8', errors='ignore')
            if isinstance(e.stderr, bytes)
            else str(e.stderr)
        )
        raise RuntimeError(f"Audio extraction failed: {stderr_msg[:200]}") from e


_original_parse_video = AsyncMultiModalContentParser.parse_video


@register_patch("NPUAsyncMultiModalContentParser", AsyncMultiModalContentParser)
class AsyncMultiModalContentParser(VLLMPatch):
    """
    Patch to modify the AsyncMultiModalContentParser behavior for compatibility with vLLM.
    """
    _attr_names_to_apply = ['parse_video', 'get_ffmpeg_input', 'audio_load_bytes']

    @staticmethod
    def get_ffmpeg_input(video_url: str, allowed_local_media_path: str) -> Tuple[str, Dict[str, Union[bytes, str]]]:
        """返回 (input_arg, run_kwargs)"""
        parsed = urlparse(video_url)
        if parsed.scheme == "data":
            # Data URL: Decode and use pipe.
            data_spec, data = parsed.path.split(",", 1)
            media_type, data_type = data_spec.split(";", 1)
            media_spec, video_format = media_type.split("/", 1)
            video_bytes = base64.b64decode(data)
            return "pipe:", {"input": video_bytes, "format": video_format}
        elif parsed.scheme == "file":
            if allowed_local_media_path is None:
                raise RuntimeError("Cannot load local files without "
                                "`--allowed-local-media-path`.")
            filepath = Path(parsed.path)
            if allowed_local_media_path not in filepath.resolve().parents:
                raise ValueError(
                    f"The file path {filepath} must be a subpath "
                    f"of `--allowed-local-media-path` {allowed_local_media_path}.")
            return str(Path(parsed.path)), {}
        else:
            # For HTTP/HTTPS or a regular path string, send directly.
            return video_url, {}

    @staticmethod
    async def audio_load_bytes(data: bytes, sampling_rate: int) -> tuple[npt.NDArray, float]:
        return librosa.load(BytesIO(data), sr=sampling_rate)

    def parse_video(self, video_url: str | None, uuid: str | None = None) -> None:
        use_audio_in_video = getattr(self._tracker._model_config.hf_config.vision_config, "use_audio_in_video", False)
        if use_audio_in_video:
            sampling_rate = self._tracker._model_config.hf_image_processor_config.get(
                "sampling_rate", DEFAULT_AUDIO_SAMPLE_RATE
            )
            video = self._connector.fetch_video_async(video_url)

            import tempfile
            allowed_local_media_path = self._connector.allowed_local_media_path
            video_input, kwargs = self.get_ffmpeg_input(video_url, allowed_local_media_path)

            if "input" in kwargs:
                # When video_url is base64 string.
                input_format = kwargs.pop("format", None)
                if not input_format or not isinstance(input_format, str):
                    raise ValueError("Missing or invalid 'format' argument for binary input.")
                safe_suffix = "".join(c for c in input_format if c.isalnum())
                if not safe_suffix:
                    raise ValueError("Invalid file format string.")
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{safe_suffix}") as tmp:
                        tmp_path = tmp.name
                        tmp.write(kwargs["input"])
                        tmp.flush()
                    audio_bytes = _run_ffmpeg_extract(tmp_path, sampling_rate)
                finally:
                    if tmp_path is not None:
                        try:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                        except OSError as cleanup_err:
                            print(f"Warning: Failed to delete temporary file {tmp_path}: {cleanup_err}")
            else:
                # When video_url is local path or HTTP/HTTPS path.
                audio_bytes = _run_ffmpeg_extract(video_input, sampling_rate)
            audio = self.audio_load_bytes(audio_bytes, sampling_rate)
            self._tracker.add("audio", audio)
            video_placeholder = self._tracker.add("video", video, uuid)
            self._add_placeholder("video", video_placeholder)
        else:
            _original_parse_video(self, video_url, uuid)


_original_gather_mm_embeddings = GPUModelRunner._gather_mm_embeddings


@register_patch("GPUModelRunnerForVideoWithAudio", GPUModelRunner)
class GPUModelRunner(VLLMPatch):
    """
    Patch to modify the GPUModelRunner behavior for compatibility with vLLM.
    """
    _attr_names_to_apply = ['_gather_mm_embeddings']

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
        shift_computed_tokens: int = 0,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        # Ensure multimodal attributes are initialized before calling original method
        if not hasattr(self, 'is_mm_embed_idx'):
            self.is_mm_embed_idx = 0
        if not hasattr(self, 'is_mm_embed_buffers'):
            self.is_mm_embed_buffers = [
                self._make_buffer(self.max_num_tokens, dtype=torch.bool),
                self._make_buffer(self.max_num_tokens, dtype=torch.bool),
            ]
        start_pos_list = [mm_feature.mm_position.offset 
                            for req_id in self.input_batch.req_ids 
                                for mm_feature in self.requests[req_id].mm_features]
        if(len(start_pos_list) > len(set(start_pos_list))):
            total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
            # Swap to the other buffer to avoid race condition with previous
            # iteration's async copy that may still be reading from CPU.
            self.is_mm_embed_idx = 1 - self.is_mm_embed_idx
            is_mm_embed_buf = self.is_mm_embed_buffers[self.is_mm_embed_idx]

            mm_embeds: list[torch.Tensor] = []
            is_mm_embed = is_mm_embed_buf.cpu
            is_mm_embed[:total_num_scheduled_tokens] = False

            req_start_idx = 0
            should_sync_mrope_positions = False
            should_sync_xdrope_positions = False

            for req_id in self.input_batch.req_ids:
                mm_embeds_req: list[torch.Tensor] = []

                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                req_state = self.requests[req_id]
                num_computed_tokens = req_state.num_computed_tokens + shift_computed_tokens

                pre_start_pos = -1
                pre_is_embed = None

                for mm_feature in req_state.mm_features:
                    pos_info = mm_feature.mm_position
                    start_pos = pos_info.offset
                    num_encoder_tokens = pos_info.length

                    # The encoder output is needed if the two ranges overlap:
                    # [num_computed_tokens,
                    #  num_computed_tokens + num_scheduled_tokens) and
                    # [start_pos, start_pos + num_encoder_tokens)
                    if start_pos >= num_computed_tokens + num_scheduled_tokens:
                        # The encoder output is not needed in this step.
                        break
                    if start_pos + num_encoder_tokens <= num_computed_tokens:
                        # The encoder output is already processed and stored
                        # in the decoder's KV cache.
                        continue

                    start_idx = max(num_computed_tokens - start_pos, 0)
                    end_idx = min(
                        num_computed_tokens - start_pos + num_scheduled_tokens,
                        num_encoder_tokens,
                    )
                    if start_idx >= end_idx:
                        raise RuntimeError(
                            f"Invalid multimodal embed range: start_idx={start_idx}, "
                            f"end_idx={end_idx}."
                        )
                    curr_embeds_start, curr_embeds_end = (
                        pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                    )
                    # If there are no embeddings in the current range, we skip
                    # gathering the embeddings.
                    if curr_embeds_start == curr_embeds_end:
                        continue

                    mm_hash = mm_feature.identifier
                    encoder_output = self.encoder_cache.get(mm_hash, None)
                    if encoder_output is None:
                        raise RuntimeError(f"Encoder cache miss for {mm_hash}.")

                    if (is_embed := pos_info.is_embed) is not None:
                        is_embed = is_embed[start_idx:end_idx]
                        if start_pos == pre_start_pos:
                            mm_embeds_item = torch.zeros(
                                (len(is_embed), encoder_output.shape[1]),
                                dtype=encoder_output.dtype,
                                device=encoder_output.device,
                            )
                            mm_embeds_item[is_embed] = encoder_output[
                                curr_embeds_start:curr_embeds_end
                            ]
                            mm_embeds_item[pre_is_embed] = mm_embeds_req[-1]
                            is_embed = is_embed | pre_is_embed
                            mm_embeds_item = mm_embeds_item[is_embed]
                            mm_embeds_req.pop(-1)
                        else:
                            mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
                    else:
                        mm_embeds_item = encoder_output[start_idx:end_idx]

                    req_start_pos = req_start_idx + start_pos - num_computed_tokens
                    is_mm_embed[req_start_pos + start_idx:req_start_pos + end_idx] = (
                        True if is_embed is None else is_embed
                    )

                    pre_start_pos = start_pos
                    pre_is_embed = is_embed
                    mm_embeds_req.append(mm_embeds_item)

                if self.is_multimodal_pruning_enabled and self.uses_mrope:
                    if req_state.mrope_positions is None:
                        raise RuntimeError("M-ROPE positions are required for multimodal pruning.")
                    should_sync_mrope_positions = True
                    mm_embeds_req, new_mrope_positions, new_delta = (
                        self.model.recompute_mrope_positions(
                            input_ids=req_state.prompt_token_ids,
                            multimodal_embeddings=mm_embeds_req,
                            mrope_positions=req_state.mrope_positions,
                            num_computed_tokens=req_state.num_computed_tokens,
                        )
                    )
                    req_state.mrope_positions.copy_(new_mrope_positions)
                    req_state.mrope_position_delta = new_delta

                mm_embeds.extend(mm_embeds_req)
                req_start_idx += num_scheduled_tokens

            is_mm_embed = is_mm_embed_buf.copy_to_gpu(total_num_scheduled_tokens)

            if should_sync_mrope_positions:
                self._calc_mrope_positions(scheduler_output)
                self.mrope_positions.copy_to_gpu(total_num_scheduled_tokens)

            if should_sync_xdrope_positions:
                self._calc_xdrope_positions(scheduler_output)
                self.xdrope_positions.copy_to_gpu(total_num_scheduled_tokens)

            return mm_embeds, is_mm_embed
        else:

            return _original_gather_mm_embeddings(self, scheduler_output, shift_computed_tokens)
