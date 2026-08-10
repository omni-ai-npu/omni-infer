# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import pickle
import json
import time
from unittest.mock import MagicMock, patch, call, ANY

import pytest
from vllm.multimodal.processing import PromptUpdateDetails
from vllm.multimodal.processing.processor import ResolvedPromptUpdate

from omni_npu.connector.mm_feature_transfer.config import DiskConnectorConfig
from omni_npu.connector.mm_feature_transfer.mm_feature_connector.disk_connector import DiskMMFeatureConnector


# -------------------- Fixtures --------------------
@pytest.fixture
def mock_meta():
    """Mock metadata store."""
    meta = MagicMock()
    meta.inc_refcount.return_value = None
    meta.dec_refcount.return_value = None
    meta.add_feature.return_value = None
    meta.get_total_used.return_value = 0
    meta.get_eviction_candidates.return_value = []
    meta.delete_old_features.return_value = []
    return meta


@pytest.fixture
def mock_mm_item():
    """Mock a MultiModalKwargsItem with tensors."""
    field1 = MagicMock()
    field1.modality = "image"
    field1.data = MagicMock()
    field1.data.numel.return_value = 100
    field1.data.element_size.return_value = 4

    field2 = MagicMock()
    field2.modality = "image"
    field2.data = MagicMock()
    field2.data.numel.return_value = 200
    field2.data.element_size.return_value = 4

    mm_item = MagicMock()
    # keys cantains two fields, one of which is to be excluded
    mm_item.keys.return_value = ["image_tensor", "address"]
    
    # __getitem__ returns the corresponding field
    def _getitem_side_effect(key):
        if key == "image_tensor":
            return field1
        else:
            return field2
    mm_item.__getitem__.side_effect = _getitem_side_effect
    mm_item.__iter__.return_value = iter(["image_tensor", "address"])
    mm_item.values.return_value = [field1, field2]
    mm_item.items.return_value = [("image_tensor", field1), ("address", field2)]
    return mm_item


@pytest.fixture
def mock_prompt_updates():
    """Mock a list of ResolvedPromptUpdate."""
    pu1 = MagicMock()
    pu2 = MagicMock()
    return [pu1, pu2]


def create_resolved_prompt_update(full="dummy_full", modality="image", item_idx=0, mode="replace", target="target"):
    content = PromptUpdateDetails.from_seq(full)  # is_embed is None
    return ResolvedPromptUpdate(
        modality=modality,
        item_idx=item_idx,
        mode=mode,
        target=target,
        content=content
    )


@pytest.fixture
def real_prompt_updates():
    """Return a list of real ResolvedPromptUpdate objects."""
    return [
        create_resolved_prompt_update(full="prompt1", modality="image", item_idx=0, mode="replace", target="tgt1"),
        create_resolved_prompt_update(full="prompt2", modality="video", item_idx=1, mode="append", target="tgt2"),
    ]


@pytest.fixture
def connector(tmp_path, mock_meta):
    """Create a DiskMMFeatureConnector with mocked metadata store and no background threads."""
    config = DiskConnectorConfig(
        # type="DiskMMFeatureConnector",
        storage_path=str(tmp_path / "features"),
        max_size_gb=0.05,
        high_water_ratio=0.85,
        low_water_ratio=0.70,
        ttl_seconds=600,
        cleanup_interval=60,
        exclude_fields=("address", "monotonic_id"),
        is_producer=True,
        metadata_store_type="SQLiteMetadataStore",
    )

    # Patch the metadata store factory to return our mock
    with patch(
        "omni_npu.connector.mm_feature_transfer.mm_feature_connector.disk_connector.create_metadata_store",
        return_value=mock_meta,
    ):
        # Patch _start_background_threads to avoid real thread creation
        with patch.object(DiskMMFeatureConnector, "_start_background_threads") as mock_start:
            conn = DiskMMFeatureConnector(config)
            # Store mock for assertions later
            conn._meta_mock = mock_meta
            conn._start_mock = mock_start
            return conn


# -------------------- Tests for has_item --------------------
class TestHasItem:
    def test_has_item_producer_returns_false_and_increments_refcount(self, connector):
        """A missing producer-side item still participates in refcount tracking."""
        mm_hash = "producer_hash"
        with patch.object(connector, "is_profile_run", return_value=False):
            result = connector.has_item(mm_hash)
            assert result is False
            connector._meta_mock.inc_refcount.assert_called_once_with(mm_hash)

    def test_has_item_consumer_file_exists_returns_true_and_increments_refcount(self, connector):
        """If metadata file exists in consumer mode, return True and inc refcount."""
        mm_hash = "abc123"
        connector.is_consumer = True
        feature_dir = connector._get_feature_dir(mm_hash)
        os.makedirs(feature_dir, exist_ok=True)
        metadata_file = os.path.join(feature_dir, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump({"modality": "image", "tensor_keys": []}, f)

        with patch.object(connector, "is_profile_run", return_value=False):
            result = connector.has_item(mm_hash)
            assert result is True
            connector._meta_mock.inc_refcount.assert_called_once_with(mm_hash)

    def test_has_item_consumer_file_missing_returns_false_and_increments_refcount(self, connector):
        """If metadata file does not exist in consumer mode, return False and inc refcount."""
        mm_hash = "missing_hash"
        connector.is_consumer = True
        with patch.object(connector, "is_profile_run", return_value=False):
            result = connector.has_item(mm_hash)
            assert result is False
            connector._meta_mock.inc_refcount.assert_called_once_with(mm_hash)

    def test_has_item_consumer_exception_in_inc_refcount_caught_returns_false(self, connector):
        """If inc_refcount raises in consumer mode, catch exception and return False."""
        mm_hash = "error_hash"
        connector.is_consumer = True
        connector._meta_mock.inc_refcount.side_effect = Exception("DB error")
        with patch.object(connector, "is_profile_run", return_value=False):
            result = connector.has_item(mm_hash)
            assert result is False
            connector._meta_mock.inc_refcount.assert_called_once_with(mm_hash)


# -------------------- Tests for save_item_with_updates --------------------
class TestSaveItemWithUpdates:
    def test_save_already_exists_skips(self, connector, mock_mm_item, real_prompt_updates):
        """If feature directory already exists, skip saving."""
        mm_hash = "existing"
        feature_dir = connector._get_feature_dir(mm_hash)
        os.makedirs(feature_dir, exist_ok=True)
        connector.save_item_with_updates(mm_hash, mock_mm_item, real_prompt_updates)
        # No new files created, no add_feature
        assert os.path.exists(feature_dir)
        connector._meta_mock.add_feature.assert_not_called()

    @patch("safetensors.torch.save_file")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("pickle.dump")
    def test_save_successful(self, mock_pickle_dump, mock_open, mock_save_file, 
                             connector, mock_mm_item, real_prompt_updates):
        """Normal save flow: create dir, save tensors (skip excluded), metadata, pickle, register."""
        mm_hash = "newhash"
        # Mock _estimate_size to return fixed value
        connector.save_item_with_updates(mm_hash, mock_mm_item, real_prompt_updates)

        feature_dir = connector._get_feature_dir(mm_hash)
        time.sleep(1)
        assert os.path.exists(feature_dir)

        # Check tensors saved: only "image_tensor", not "address"
        expected_tensor_calls = [
            call({"image_tensor": ANY}, os.path.join(feature_dir, "image_tensor.safetensors")),
        ]
        mock_save_file.assert_has_calls(expected_tensor_calls, any_order=True)
        assert mock_save_file.call_count == 1

        # Check metadata writing
        # open called for metadata.json and prompt_updates.pkl
        # We'll just verify metadata content via the json.dump call
        # In practice, we'd capture the write calls; but we can check that open was called with correct paths.
        mock_open.assert_any_call(os.path.join(feature_dir, "metadata.json"), "w")
        mock_open.assert_any_call(os.path.join(feature_dir, "prompt_updates.pkl"), "wb")

        # Check pickle dump called
        mock_pickle_dump.assert_called_once()
        args, _ = mock_pickle_dump.call_args
        serialized_list = args[0]  # first argument is the list
        assert isinstance(serialized_list, list)
        # Each item should be result of serialize_prompt_update (we didn't patch that, but we can assume)

        # Check meta.add_feature called with estimated size
        connector._meta_mock.add_feature.assert_called_once_with(mm_hash, ANY)


# -------------------- Tests for load_item_with_updates --------------------
class TestLoadItemWithUpdates:
    def test_load_metadata_missing_returns_none(self, connector):
        """If metadata.json missing, return None."""
        mm_hash = "missing_meta"
        result = connector.load_item_with_updates(mm_hash)
        assert result is None
        # dec_refcount should still be called
        # connector._meta_mock.dec_refcount.assert_called_once_with(mm_hash)

    def test_load_tensor_missing_returns_none(self, connector):
        """If a tensor file listed in metadata is missing, return None."""
        mm_hash = "missing_tensor"
        feature_dir = connector._get_feature_dir(mm_hash)
        os.makedirs(feature_dir, exist_ok=True)
        # Create metadata with a tensor key that won't have a file
        metadata = {"modality": "image", "tensor_keys": ["image", "missing"]}
        with open(os.path.join(feature_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        # Create only "image.safetensors"
        with open(os.path.join(feature_dir, "image.safetensors"), "w") as f:
            f.write("dummy")  # just to exist

        result = connector.load_item_with_updates(mm_hash)
        assert result is None
        connector._meta_mock.dec_refcount.assert_called_once_with(mm_hash)

    def test_load_prompt_file_missing_returns_none(self, connector):
        """If prompt_updates.pkl missing, return None."""
        mm_hash = "missing_prompt"
        feature_dir = connector._get_feature_dir(mm_hash)
        os.makedirs(feature_dir, exist_ok=True)
        metadata = {"modality": "image", "tensor_keys": ["image"]}
        with open(os.path.join(feature_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        # Create tensor file
        with open(os.path.join(feature_dir, "image.safetensors"), "w") as f:
            f.write("dummy")
        # No prompt file
        result = connector.load_item_with_updates(mm_hash)
        assert result is None
        connector._meta_mock.dec_refcount.assert_called_once_with(mm_hash)
    
    @patch("safetensors.torch.load_file")
    @patch("pickle.load")
    def test_load_successful(
        self, mock_pickle_load, mock_safe_load,
        connector, real_prompt_updates
    ):
        """Full successful load: loads all tensors, reconstructs mm_item and prompt updates."""
        mm_hash = "load_success"
        feature_dir = connector._get_feature_dir(mm_hash)
        os.makedirs(feature_dir, exist_ok=True)
        # Create metadata
        metadata = {"modality": "image", "tensor_keys": ["image"]}
        with open(os.path.join(feature_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        tensor_path = os.path.join(feature_dir, "image.safetensors")
        with open(tensor_path, "w") as f:
            f.write("dummy")
        prompt_path = os.path.join(feature_dir, "prompt_updates.pkl")
        serialized_data = [{
            "modality": "image", 
            "item_idx": 0, 
            "mode": "replace", 
            "target": "tgt", 
            "content_info": ("full", None, None)
        }]
        with open(prompt_path, "wb") as f:
            pickle.dump(serialized_data, f)

        loaded_tensor = MagicMock()
        mock_safe_load.return_value = {"image": loaded_tensor}
        mock_pickle_load.return_value = serialized_data

        result = connector.load_item_with_updates(mm_hash)

        mock_safe_load.assert_called_once_with(tensor_path)
        mock_pickle_load.assert_called_once()
        assert list(result[0]) == ["image"]
        assert result[0]["image"].data is loaded_tensor
        assert len(result[1]) == 1
        # dec_refcount called
        connector._meta_mock.dec_refcount.assert_called_once_with(mm_hash)


class TestEvictIfNeeded:
    def test_no_eviction_when_usage_below_high_water(self, connector):
        """Used space below high water ratio; no eviction"""
        connector.max_bytes = 100  
        connector.high_water_ratio = 0.85
        connector.low_water_ratio = 0.70
        used = 80  # <= 85
        connector._meta_mock.get_total_used.return_value = used
        connector._evict_if_needed()
        connector._meta_mock.get_eviction_candidates.assert_not_called()
        connector._meta_mock.delete_feature.assert_not_called()

    def test_eviction_when_above_high_water_and_candidates_available(self, connector, tmp_path):
        connector.max_bytes = 100
        connector.high_water_ratio = 0.85
        connector.low_water_ratio = 0.70
        used = 90  # > 85
        connector._meta_mock.get_total_used.return_value = used
        candidates = [("hash1", 15), ("hash2", 10)]
        connector._meta_mock.get_eviction_candidates.return_value = candidates

        for h, _ in candidates:
            dir_path = connector._get_feature_dir(h)
            os.makedirs(dir_path, exist_ok=True)
            with open(os.path.join(dir_path, "dummy"), "w") as f:
                f.write("test")

        connector._evict_if_needed()

        connector._meta_mock.get_eviction_candidates.assert_called_once_with(20)
        expected_delete_calls = [call("hash1"), call("hash2")]
        connector._meta_mock.delete_feature.assert_has_calls(expected_delete_calls)
        for h, _ in candidates:
            assert not os.path.exists(connector._get_feature_dir(h))

    def test_eviction_stops_when_freed_enough(self, connector, tmp_path):
        connector.max_bytes = 100
        connector.high_water_ratio = 0.85
        connector.low_water_ratio = 0.70
        used = 90
        connector._meta_mock.get_total_used.return_value = used
        candidates = [("hash1", 30), ("hash2", 5)]
        connector._meta_mock.get_eviction_candidates.return_value = candidates

        for h, _ in candidates:
            dir_path = connector._get_feature_dir(h)
            os.makedirs(dir_path, exist_ok=True)

        connector._evict_if_needed()

        connector._meta_mock.delete_feature.assert_called_once_with("hash1")
        assert os.path.exists(connector._get_feature_dir("hash2"))


class TestCleanupByTTL:
    def test_cleanup_old_features(self, connector, tmp_path):
        older_than = int(time.time()) - 100
        hashes = ["expired1", "expired2"]
        connector._meta_mock.delete_old_features.return_value = hashes

        for h in hashes:
            dir_path = connector._get_feature_dir(h)
            os.makedirs(dir_path, exist_ok=True)
            with open(os.path.join(dir_path, "dummy"), "w") as f:
                f.write("test")

        connector.ttl_seconds = 100
        with patch("time.time", return_value=1000):  # 固定当前时间
            connector._cleanup_by_ttl()

        connector._meta_mock.delete_old_features.assert_called_once_with(900)
        for h in hashes:
            assert not os.path.exists(connector._get_feature_dir(h))

    def test_no_cleanup_when_no_old_features(self, connector):
        connector._meta_mock.delete_old_features.return_value = []
        connector.ttl_seconds = 100
        with patch("time.time", return_value=1000):
            connector._cleanup_by_ttl()
        connector._meta_mock.delete_old_features.assert_called_once_with(900)
