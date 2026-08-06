# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import threading
import sqlite3
import time
from unittest.mock import MagicMock, patch

import pytest

from omni.connector.mm_feature_transfer.config import DiskConnectorConfig
from omni.connector.mm_feature_transfer.mm_feature_connector.disk_connector import (
    DiskMMFeatureConnector,
    SQLiteMetadataStore,
)


DISK_CONNECTOR_MODULE = DiskMMFeatureConnector.__module__


def _create_legacy_sqlite_db(db_path: str, mm_hash: str = "legacy_hash") -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS features (
            mm_hash TEXT PRIMARY KEY,
            size INTEGER NOT NULL,
            refcount INTEGER NOT NULL DEFAULT 0,
            last_used INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO features (mm_hash, size, refcount, last_used) VALUES (?, ?, ?, ?)",
        (mm_hash, 123, 0, 1),
    )
    conn.commit()
    conn.close()


@pytest.mark.unit
class TestSQLiteMetadataStoreStartupReset:
    def test_cleanup_existing_true_clears_previous_db_state(self, tmp_path):
        db_path = tmp_path / "metadata.db"
        _create_legacy_sqlite_db(str(db_path))

        wal_path = f"{db_path}-wal"
        shm_path = f"{db_path}-shm"
        for sidecar_path in (wal_path, shm_path):
            with open(sidecar_path, "w", encoding="utf-8") as handle:
                handle.write("stale")

        store = SQLiteMetadataStore(str(db_path), cleanup_existing=True)
        try:
            assert store.get_refcount("legacy_hash") is None
        finally:
            store.close()

    def test_cleanup_existing_false_preserves_previous_db_state(self, tmp_path):
        db_path = tmp_path / "metadata.db"
        _create_legacy_sqlite_db(str(db_path))

        store = SQLiteMetadataStore(str(db_path), cleanup_existing=False)
        try:
            assert store.get_refcount("legacy_hash") == 0
        finally:
            store.close()

    def test_context_manager_and_close_are_idempotent(self, tmp_path):
        db_path = tmp_path / "metadata.db"
        with SQLiteMetadataStore(str(db_path)) as store:
            store.add_feature("ctx_hash", 1)
            assert store.get_refcount("ctx_hash") == 0

        store.close()
        store.close()

    def test_cleanup_existing_true_recreates_clean_db(self, tmp_path):
        db_path = tmp_path / "metadata.db"
        _create_legacy_sqlite_db(str(db_path))

        with patch.object(SQLiteMetadataStore, "_current_mode", return_value="wal"):
            store = SQLiteMetadataStore(str(db_path), cleanup_existing=True)
        try:
            assert store.get_refcount("legacy_hash") is None
        finally:
            store.close()

    def test_unknown_store_type_raises_value_error(self):
        from omni.connector.mm_feature_transfer.mm_feature_connector.disk_connector import (
            create_metadata_store,
        )

        with pytest.raises(ValueError):
            create_metadata_store("UnknownStore", cleanup_existing=True)


@pytest.mark.unit
class TestDiskMMFeatureConnectorStartupReset:
    def test_producer_init_resets_storage_path_and_passes_cleanup_flag(self, tmp_path):
        storage_path = tmp_path / "features"
        nested_dir = storage_path / "old_feature"
        nested_dir.mkdir(parents=True)
        stale_file = nested_dir / "metadata.json"
        stale_file.write_text("stale", encoding="utf-8")
        loose_file = storage_path / "stale.bin"
        loose_file.parent.mkdir(parents=True, exist_ok=True)
        loose_file.write_text("stale", encoding="utf-8")

        config = DiskConnectorConfig(
            storage_path=str(storage_path),
            max_size_gb=0.05,
            high_water_ratio=0.85,
            low_water_ratio=0.70,
            ttl_seconds=600,
            cleanup_interval=60,
            exclude_fields=("address", "monotonic_id"),
            is_producer=True,
            metadata_store_type="SQLiteMetadataStore",
        )

        mock_meta = MagicMock()
        with patch(f"{DISK_CONNECTOR_MODULE}.create_metadata_store", return_value=mock_meta) as mock_create_meta:
            with patch.object(DiskMMFeatureConnector, "_start_background_threads") as mock_start_threads:
                connector = DiskMMFeatureConnector(config)

        try:
            mock_create_meta.assert_called_once_with("SQLiteMetadataStore", True)
            mock_start_threads.assert_called_once()
            assert storage_path.exists()
            assert not stale_file.exists()
            assert not nested_dir.exists()
            assert not loose_file.exists()
        finally:
            connector.close()

    def test_consumer_init_keeps_storage_path_contents(self, tmp_path):
        storage_path = tmp_path / "features"
        nested_dir = storage_path / "old_feature"
        nested_dir.mkdir(parents=True)
        stale_file = nested_dir / "metadata.json"
        stale_file.write_text("stale", encoding="utf-8")

        config = DiskConnectorConfig(
            storage_path=str(storage_path),
            max_size_gb=0.05,
            high_water_ratio=0.85,
            low_water_ratio=0.70,
            ttl_seconds=600,
            cleanup_interval=60,
            exclude_fields=("address", "monotonic_id"),
            is_producer=False,
            metadata_store_type="SQLiteMetadataStore",
        )

        mock_meta = MagicMock()
        with patch(f"{DISK_CONNECTOR_MODULE}.create_metadata_store", return_value=mock_meta) as mock_create_meta:
            connector = DiskMMFeatureConnector(config)

        try:
            mock_create_meta.assert_called_once_with("SQLiteMetadataStore", False)
            assert stale_file.exists()
            assert nested_dir.exists()
        finally:
            connector.close()

    def test_close_stops_worker_and_closes_metadata(self, tmp_path):
        storage_path = tmp_path / "features"
        storage_path.mkdir(parents=True)

        config = DiskConnectorConfig(
            storage_path=str(storage_path),
            max_size_gb=0.05,
            high_water_ratio=0.85,
            low_water_ratio=0.70,
            ttl_seconds=600,
            cleanup_interval=60,
            exclude_fields=("address", "monotonic_id"),
            is_producer=True,
            metadata_store_type="SQLiteMetadataStore",
        )

        mock_meta = MagicMock()
        with patch(f"{DISK_CONNECTOR_MODULE}.create_metadata_store", return_value=mock_meta):
            with patch.object(DiskMMFeatureConnector, "_start_background_threads"):
                connector = DiskMMFeatureConnector(config)

        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        connector._cleanup_thread = mock_thread
        connector.close()
        connector.close()

        assert connector._shutdown_event.is_set()
        mock_thread.join.assert_called_once()
        mock_meta.close.assert_called_once()

    def test_reset_storage_path_removes_root_file_and_directory(self, tmp_path):
        storage_path = tmp_path / "features"
        storage_path.mkdir(parents=True)
        root_file = storage_path / "stale.bin"
        root_file.write_text("stale", encoding="utf-8")
        nested_dir = storage_path / "nested"
        nested_dir.mkdir()
        (nested_dir / "inner.txt").write_text("inner", encoding="utf-8")

        config = DiskConnectorConfig(
            storage_path=str(storage_path),
            max_size_gb=0.05,
            high_water_ratio=0.85,
            low_water_ratio=0.70,
            ttl_seconds=600,
            cleanup_interval=60,
            exclude_fields=("address", "monotonic_id"),
            is_producer=True,
            metadata_store_type="SQLiteMetadataStore",
        )

        mock_meta = MagicMock()
        with patch(f"{DISK_CONNECTOR_MODULE}.create_metadata_store", return_value=mock_meta):
            with patch.object(DiskMMFeatureConnector, "_start_background_threads"):
                connector = DiskMMFeatureConnector(config)

        try:
            assert not root_file.exists()
            assert not nested_dir.exists()
        finally:
            connector.close()