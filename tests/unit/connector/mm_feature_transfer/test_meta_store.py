# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import time
import pytest
from unittest.mock import patch

from omni.connector.mm_feature_transfer.mm_feature_connector.disk_connector import SQLiteMetadataStore


# ---------- Test SQLiteMetadataStore ----------
class TestSQLiteMetadataStore:
    @pytest.fixture
    def store(self):
        db = SQLiteMetadataStore(":memory:")
        yield db
        db.conn.close()

    def test_add_feature(self, store):
        store.add_feature("hash1", 100)
        cur = store.conn.execute("SELECT size, refcount, last_used FROM features WHERE mm_hash='hash1'")
        row = cur.fetchone()
        assert row is not None
        assert row[0] == 100
        assert row[1] == 0
        assert row[2] > 0  

    def test_inc_refcount(self, store):
        store.add_feature("h1", 50)
        store.inc_refcount("h1")
        cur = store.conn.execute("SELECT refcount FROM features WHERE mm_hash='h1'")
        assert cur.fetchone()[0] == 1
        store.inc_refcount("h1")
        cur = store.conn.execute("SELECT refcount FROM features WHERE mm_hash='h1'")
        assert cur.fetchone()[0] == 2

    def test_dec_refcount(self, store):
        store.add_feature("h1", 10)
        store.inc_refcount("h1")
        store.inc_refcount("h1")
        store.dec_refcount("h1")
        cur = store.conn.execute("SELECT refcount FROM features WHERE mm_hash='h1'")
        assert cur.fetchone()[0] == 1
        store.dec_refcount("h1")
        cur = store.conn.execute("SELECT refcount FROM features WHERE mm_hash='h1'")
        assert cur.fetchone()[0] == 0
        store.dec_refcount("h1")  # refcount 为0时不减
        cur = store.conn.execute("SELECT refcount FROM features WHERE mm_hash='h1'")
        assert cur.fetchone()[0] == 0

    def test_get_refcount(self, store):
        assert store.get_refcount("missing") is None
        store.add_feature("h1", 5)
        assert store.get_refcount("h1") == 0
        store.inc_refcount("h1")
        assert store.get_refcount("h1") == 1

    def test_get_total_used(self, store):
        assert store.get_total_used() == 0
        store.add_feature("h1", 10)
        store.add_feature("h2", 20)
        assert store.get_total_used() == 30

    def test_get_eviction_candidates(self, store):
        store.add_feature("a", 100)
        time.sleep(0.01)
        store.add_feature("b", 200)
        time.sleep(0.01)
        store.add_feature("c", 300)
        store.inc_refcount("b")

        candidates = store.get_eviction_candidates(150)
        assert candidates == [("a", 100), ("c", 300)]  # Total size reaches 400，> 150

        candidates = store.get_eviction_candidates(50)
        assert candidates == [("a", 100)]

        candidates = store.get_eviction_candidates(1000)
        assert candidates == [("a", 100), ("c", 300)]

    def test_delete_feature(self, store):
        store.add_feature("h1", 10)
        store.delete_feature("h1")
        assert store.get_refcount("h1") is None

    def test_delete_old_features(self, store):
        store.add_feature("old", 10)
        time.sleep(1)
        now = int(time.time())
        store.add_feature("new", 20)
        deleted = store.delete_old_features(now)
        assert deleted == ["old"]
        assert store.get_refcount("old") is None
        assert store.get_refcount("new") == 0

        deleted = store.delete_old_features(now)
        assert deleted == []

    def test_delete_old_features_skips_refcount_positive(self, store):
        store.add_feature("h1", 10)
        store.inc_refcount("h1")
        old_time = int(time.time()) - 100
        store.conn.execute("UPDATE features SET last_used = ? WHERE mm_hash='h1'", (old_time,))
        store.conn.commit()
        deleted = store.delete_old_features(old_time + 10)
        assert deleted == []  
        assert store.get_refcount("h1") == 1