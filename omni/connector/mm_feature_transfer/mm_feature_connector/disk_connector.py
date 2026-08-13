# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Sequence
import atexit
import json
import os
import pickle
from typing import List, Tuple, Dict, Any, Callable, TypeVar
import torch
import threading
import time
import shutil
import sqlite3
import random
import safetensors.torch

from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalKwargsItem

from ..config import ConnectorConfig
from .base import BaseMMFeatureConnector


logger = init_logger(__name__)

T = TypeVar("T")


class MMFeatureMetadataStore(ABC):
    """Abstract interface for storing and managing MM feature metadata."""
    
    @abstractmethod
    def add_feature(self, mm_hash: str, size: int) -> None:
        """Register a new feature with initial refcount=0 and last_used=now."""
        pass

    @abstractmethod
    def inc_refcount(self, mm_hash: str) -> None:
        """Increase reference count and update last_used timestamp."""
        pass

    @abstractmethod
    def dec_refcount(self, mm_hash: str) -> None:
        """Decrease reference count and update last_used timestamp."""
        pass

    @abstractmethod
    def get_refcount(self, mm_hash: str) -> int:
        """Get current reference count for a feature."""
        pass

    @abstractmethod
    def get_total_used(self) -> int:
        """Return total bytes used by all features."""
        pass

    @abstractmethod
    def get_eviction_candidates(self, required_bytes: int) -> List[Tuple[str, int]]:
        """
        Return list of (mm_hash, size) with refcount=0, sorted by last_used ASC.
        The list is truncated when cumulative size >= required_bytes.
        """
        pass

    @abstractmethod
    def delete_feature(self, mm_hash: str) -> None:
        """Remove feature from metadata store."""
        pass

    @abstractmethod
    def delete_old_features(self, older_than: int) -> List[str]:
        """
        Delete all features with refcount=0 and last_used < older_than.
        Returns list of deleted mm_hash.
        """
        pass


class SQLiteMetadataStore(MMFeatureMetadataStore):
    def __init__(self, db_path: str, cleanup_existing: bool = True) -> None:
        self.db_path = db_path
        self._db_lock = threading.RLock()
        self._closed = False
        self._max_retry_attempts = 5
        self._retry_base_sleep = 0.05
        self._max_retry_total_seconds = 3.0
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=5.0)
        self.conn.execute("PRAGMA busy_timeout=5000")  # 5 seconds
        self._ensure_wal_mode()
        if cleanup_existing:
            self._init_tables()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        with self._db_lock:
            if self._closed:
                return
            self.conn.close()
            self._closed = True

    def add_feature(self, mm_hash: str, size: int) -> None:
        self._run_with_retry(lambda: self.conn.execute(
            "INSERT INTO features (mm_hash, size, refcount, last_used) VALUES (?, ?, 0, ?)",
            (mm_hash, size, self._now())
        ))
        self._run_with_retry(self.conn.commit)

    def inc_refcount(self, mm_hash: str) -> None:
        self._run_with_retry(lambda: self.conn.execute(
            "UPDATE features SET refcount = refcount + 1, last_used = ? WHERE mm_hash = ?",
            (self._now(), mm_hash)
        ))
        self._run_with_retry(self.conn.commit)

    def dec_refcount(self, mm_hash: str) -> None:
        self._run_with_retry(lambda: self.conn.execute(
            "UPDATE features SET refcount = refcount - 1, last_used = ? WHERE mm_hash = ? AND refcount > 0",
            (self._now(), mm_hash)
        ))
        self._run_with_retry(self.conn.commit)

    def get_refcount(self, mm_hash: str) -> int | None:
        cur = self._run_with_retry(
            lambda: self.conn.execute("SELECT refcount FROM features WHERE mm_hash = ?", (mm_hash,))
        )
        row = self._run_with_retry(cur.fetchone)
        return row[0] if row else None

    def get_total_used(self) -> int:
        cur = self._run_with_retry(
            lambda: self.conn.execute("SELECT COALESCE(SUM(size), 0) FROM features")
        )
        return self._run_with_retry(cur.fetchone)[0]

    def get_eviction_candidates(self, required_bytes: int) -> List[Tuple[str, int]]:
        candidates = []
        cur = self._run_with_retry(lambda: self.conn.execute(
            "SELECT mm_hash, size FROM features WHERE refcount = 0 ORDER BY last_used ASC"
        ))
        total = 0
        for mm_hash, size in self._run_with_retry(cur.fetchall):
            candidates.append((mm_hash, size))
            total += size
            if total >= required_bytes:
                break
        return candidates

    def delete_feature(self, mm_hash: str) -> None:
        self._run_with_retry(lambda: self.conn.execute(
            "DELETE FROM features WHERE mm_hash = ?", (mm_hash,)
        ))
        self._run_with_retry(self.conn.commit)

    def delete_old_features(self, older_than: int) -> List[str]:
        cur = self._run_with_retry(lambda: self.conn.execute(
            "SELECT mm_hash FROM features WHERE refcount = 0 AND last_used < ?",
            (older_than,)
        ))
        hashes = [row[0] for row in self._run_with_retry(cur.fetchall)]
        if hashes:
            self._run_with_retry(lambda: self.conn.execute(
                "DELETE FROM features WHERE refcount = 0 AND last_used < ?",
                (older_than,)
            ))
            self._run_with_retry(self.conn.commit)
        return hashes

    def _init_tables(self):
        def _execute_init_logic():
            cursor = self.conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    mm_hash TEXT PRIMARY KEY,
                    size INTEGER NOT NULL,
                    refcount INTEGER NOT NULL DEFAULT 0,
                    last_used INTEGER NOT NULL
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_evict ON features(refcount, last_used)"
            )
            cursor.execute("DELETE FROM features")
        self._run_with_retry(lambda: self.conn.execute("BEGIN IMMEDIATE TRANSACTION;"))
        self._run_with_retry(_execute_init_logic)
        self._run_with_retry(self.conn.commit)

    def _current_mode(self) -> str:
        cursor = self._run_with_retry(self.conn.cursor)
        self._run_with_retry(lambda: cursor.execute("PRAGMA journal_mode"))
        current_mode = self._run_with_retry(cursor.fetchone)[0]
        return current_mode

    def _ensure_wal_mode(self) -> None:
        mode = self._current_mode().lower()
        if mode == "wal":
            return
        self._run_with_retry(lambda: self.conn.execute("PRAGMA journal_mode=WAL"))

    def _run_with_retry(self, func: Callable[[], T]) -> T:  # pylint: disable=inconsistent-return-statements
        start = time.monotonic()
        for attempt in range(self._max_retry_attempts):
            try:
                with self._db_lock:
                    return func()
            except sqlite3.OperationalError as e:
                message = str(e).lower()
                if "locked" not in message and "busy" not in message and "disk i/o error" not in message:
                    raise
                if attempt == self._max_retry_attempts - 1:
                    raise
                if time.monotonic() - start >= self._max_retry_total_seconds:
                    raise
                backoff = self._retry_base_sleep * (2 ** attempt)
                jittered_backoff = backoff * random.uniform(0.5, 1.5)
                time.sleep(min(jittered_backoff, self._max_retry_total_seconds))
        raise RuntimeError("SQLite retry loop exited unexpectedly")

    def _now(self) -> int:
        return int(time.time())


def create_metadata_store(metadata_store_type: str, cleanup_existing: bool = True) -> MMFeatureMetadataStore:
    if metadata_store_type == "SQLiteMetadataStore":
        db_path = "/dev/shm/metadata.db"
        return SQLiteMetadataStore(db_path, cleanup_existing)
    else:
        raise ValueError(f"Unknown metadata store type: {metadata_store_type}")


class DiskMMFeatureConnector(BaseMMFeatureConnector):
    """
    Disk-based MM feature connector.

    This connector persists multimodal features to disk using safetensors format.
    It supports:
    - Individual tensor granularity (each tensor saved separately)
    - Metadata stored as JSON alongside tensor files

    Stores features as individual tensors in safetensors format:
        /{storage_path}/{mm_hash}/
            ├── metadata.json          # modality, tensor_keys
            ├── prompt_updates.pkl     
            ├── image_grid_thw.safetensors   
            └── ...                    # additional tensors
    """

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)

        self.storage_path = config.storage_path
        
        # Capacity limits
        self.max_size_gb = config.max_size_gb
        self.high_water_ratio = config.high_water_ratio
        self.low_water_ratio = config.low_water_ratio
        self.max_bytes = config.max_size_gb * (1024 ** 3)

        # TTL settings
        self.ttl_seconds = config.ttl_seconds
        self.cleanup_interval = config.cleanup_interval

        # Metadata store (SQLite by default)
        self.meta = create_metadata_store(config.metadata_store_type, self.is_producer)

        # Control flags
        self._shutdown_event = threading.Event()
        self._cleanup_thread: threading.Thread | None = None
        self._closed = False

        # Ensure resources are released on normal interpreter shutdown.
        atexit.register(self.close)
        
        if self.is_producer:
            self._reset_storage_path()
            os.makedirs(self.storage_path, exist_ok=True)
            self._start_background_threads()

        logger.info(
            "DiskMMFeatureConnector initialized with storage_path=%s, "
            "is_producer=%s, max=%.2fGB, high=%.2f, low=%.2f, ttl=%ds",
            self.storage_path, self.is_producer, self.max_size_gb, 
            self.high_water_ratio, self.low_water_ratio, self.ttl_seconds
        )

    def has_item(self, mm_hash: str) -> bool:
        """Check if feature exists on disk."""
        try:
            if not self.is_consumer:
                logger.debug("Disable MMFeatureConnector loading in non consumer.")
                return False

            if self.is_profile_run():
                logger.debug("Disable MMFeatureConnector during profile run.")
                return False

            self.meta.inc_refcount(mm_hash)

            feature_dir = self._get_feature_dir(mm_hash)
            metadata_file = os.path.join(feature_dir, "metadata.json")
            exists = os.path.exists(metadata_file)
        
            logger.debug("MM feature %s exists on disk: %s", mm_hash, exists)
            return exists
        except Exception as e:
            logger.error("Failed to check connector for item %s: %s", mm_hash, e)
            return False

    async def save(
        self, 
        mm_hash: str, 
        metadata: Dict[str, Any],
        tensors: Dict[str, torch.Tensor], 
        serialized_updates: List[Dict[str, Any]],
    ) -> None:
        feature_dir = self._get_feature_dir(mm_hash)
        if os.path.exists(feature_dir):
            return
        os.makedirs(feature_dir, exist_ok=True)

        try:
            estimated_size = self._estimate_size(tensors, serialized_updates)

            for key, tensor in tensors.items():
                tensor_file = os.path.join(feature_dir, f"{key}.safetensors")
                safetensors.torch.save_file({key: tensor}, tensor_file)

            with open(os.path.join(feature_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f)

            with open(os.path.join(feature_dir, "prompt_updates.pkl"), "wb") as f:
                pickle.dump(serialized_updates, f)

            self.meta.add_feature(mm_hash, estimated_size)

            logger.debug(
                "Saved feature and prompt updates for %s (modality: %s, %d tensors, size=%d bytes)",
                mm_hash, metadata["modality"], len(tensors), estimated_size
            )
        except Exception as e:
            logger.error("Failed to save MM feature %s to connector: %s", mm_hash, e)

    def load(self, mm_hash: str):
        feature_dir = os.path.join(self.storage_path, mm_hash)
        metadata_file = os.path.join(feature_dir, "metadata.json")
        if not os.path.exists(metadata_file):
            return None

        try:
            with open(metadata_file) as f:
                metadata = json.load(f)

            tensors = {}
            for key in metadata["tensor_keys"]:
                tensor_file = os.path.join(feature_dir, f"{key}.safetensors")
                if not os.path.exists(tensor_file):
                    return None
                tensors[key] = safetensors.torch.load_file(tensor_file)[key]

            with open(os.path.join(feature_dir, "prompt_updates.pkl"), "rb") as f:
                serialized_updates = pickle.load(f)

            logger.debug("MM feature %s raw data loaded from disk", mm_hash)
            return metadata, tensors, serialized_updates
        except Exception as e:
            logger.error("Failed to load MM feature %s from connector: %s", mm_hash, e)
            return None
        finally:
            self.meta.dec_refcount(mm_hash)

    def _get_feature_dir(self, mm_hash: str) -> str:
        """Get directory path for a feature."""
        return os.path.join(self.storage_path, mm_hash)

    def _estimate_size(self, 
        tensors: Dict[str, torch.Tensor],  # Or list of tensors
        serialized_state: List[Dict[str, Any]],
    ) -> int:
        """Estimate the disk size of the feature (tensors + metadata + pickle)."""
        total = 0

        for _key, tensor in tensors.items():
            total += tensor.numel() * tensor.element_size()
        total += len(pickle.dumps(serialized_state))
        total += 1024 # metadata json (rough)

        return total

    # -------------------- Background cleanup --------------------
    
    def _start_background_threads(self):
        # Thread for regular capacity & TTL cleanup
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_worker(self):
        while not self._shutdown_event.is_set():
            try:
                # Do a capacity check and eviction if needed
                self._evict_if_needed()
                # TTL cleanup (normal)
                self._cleanup_by_ttl()
            except Exception as e:
                logger.exception("Background cleanup error: %s", e)
            time.sleep(self.cleanup_interval)
    
    def _evict_if_needed(self):
        """Check disk usage and evict refcount=0 items if above high watermark."""
        used = self.meta.get_total_used()
        if used <= self.max_bytes * self.high_water_ratio:
            return
        
        # Need to evict down to low water mark
        target_bytes = self.max_bytes * self.low_water_ratio
        to_free = used - target_bytes
        if to_free <= 0:
            return
        
        candidates = self.meta.get_eviction_candidates(to_free)
        if not candidates:
            logger.warning("Disk usage high (%.1f%%), but no evictable items (all in use)", 
                           100.0 * used / self.max_bytes)
            return
        
        freed = 0
        for mm_hash, size in candidates:
            dir_path = self._get_feature_dir(mm_hash)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path, ignore_errors=True)
            self.meta.delete_feature(mm_hash)
            freed += size
            if freed >= to_free:
                break
        logger.info("Evicted %d features, freed %.2f MB", len(candidates), freed / (1024 * 1024))

    def _cleanup_by_ttl(self):
        now = int(time.time())
        older_than = now - self.ttl_seconds
        hashes = self.meta.delete_old_features(older_than)
        for mm_hash in hashes:
            dir_path = self._get_feature_dir(mm_hash)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path, ignore_errors=True)
        if hashes:
            logger.info("TTL cleanup removed %d old features", len(hashes))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        self._shutdown_event.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=self.cleanup_interval + 1)

        self.meta.close()

    def __del__(self):
        self.close()

    def _reset_storage_path(self) -> None:
        if os.path.isdir(self.storage_path):
            for entry in os.listdir(self.storage_path):
                entry_path = os.path.join(self.storage_path, entry)
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path, ignore_errors=True)
                else:
                    try:
                        os.remove(entry_path)
                    except OSError:
                        pass

