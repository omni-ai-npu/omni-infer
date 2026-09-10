# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch
import time
from unittest.mock import patch, MagicMock

MODULE = "omni_npu.connector.utils"


def test_get_local_ip():
    from omni_npu.connector.utils import get_local_ip
    # This function connects to an external address. Mocking socket is complex here.
    # A simple test to ensure it returns a string-like IP.
    ip = get_local_ip()
    assert isinstance(ip, str)
    assert len(ip) > 0
    # Basic check for IPv4 format (not perfect, but a start)
    assert len(ip.split(".")) == 4

def test_get_kv_port():
    from omni_npu.connector.utils import get_kv_port
    def get_world_group():
        return MagicMock(local_rank=7)
    def make_config(port: int):
        return MagicMock(kv_transfer_config=MagicMock(kv_port=port))
    with patch(MODULE + ".get_world_group", get_world_group):
        assert get_kv_port(make_config(1000)) == 1007 # 1000 + 7
        assert get_kv_port(make_config(None)) == 5575 # 5568 + 7

def test_get_kv_role():
    from omni_npu.connector.utils import get_kv_role
    def make_config(role: str):
        return MagicMock(kv_transfer_config=MagicMock(kv_role=role))
    assert get_kv_role(make_config("kv_producer")) == True
    assert get_kv_role(make_config("kv_consumer")) == False
    with pytest.raises(Exception):
        get_kv_role(make_config("unknown"))

def test_serial_brief():
    from omni_npu.connector.utils import serial_brief
    s1 = [1, 2, 3, 4]
    s2 = [9, 10, 11, 12]
    s3 = [1001, 1002, 1003, 1004]
    s4 = [-1, 0, 1, 2]
    s5 = [5, 4, 3, 2]
    s6 = [1, 1, 1, 1]
    assert serial_brief(s1 + s2 + s3 + s4) == "[1~4, 9~12, 1001~1004, -1~2]"
    assert serial_brief(s2 + s4 + s5) == "[9~12, -1~2, 5, 4, 3, 2]"
    assert serial_brief(s2 + s2 + s2) == "[9~12, 9~12, 9~12]"
    assert serial_brief(s4 + s5 + s6) == "[-1~2, 5, 4, 3, 2, 1, 1, 1, 1]"

def test_calm_down():
    from omni_npu.connector.utils import calm_down
    t1 = time.time()
    calm_down("case1", 0.5)
    t2 = time.time()
    calm_down("case1", 0.5)
    t3 = time.time()
    calm_down("case2", 0.5)
    t4 = time.time()
    calm_down("case2", 0.5)
    t5 = time.time()
    assert t2 - t1 < 0.1
    assert t3 - t2 > 0.4
    assert t4 - t3 < 0.1
    assert t5 - t4 > 0.4



from omni_npu.connector.utils import TP_Convertor

class TestTPConvertor:
    """Test TP_Convertor class with various scenarios"""

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_init(self, mock_get_tp_group):
        """Test TP_Convertor initialization"""
        # Mock TP group
        mock_tp_group = MagicMock()
        mock_tp_group.device_group = "mock_comm"
        mock_tp_group.world_size = 8
        mock_tp_group.rank_in_group = 2
        mock_get_tp_group.return_value = mock_tp_group

        # Test valid initialization
        convertor = TP_Convertor(remote_tp_size=2)
        assert convertor.tp_comm == "mock_comm"
        assert convertor.tp_size == 8
        assert convertor.tp_rank == 2
        assert convertor.remote_tp_size == 2
        assert convertor.stride == 4  # 8 / 2
        assert convertor.offset == 2 % 4
        assert convertor.transfer_done is False

        # Test ValueError when tp_size % remote_tp_size != 0
        mock_tp_group.world_size = 7
        with pytest.raises(ValueError):
            TP_Convertor(remote_tp_size=2)

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_scheme_reorg_stride_1(self, mock_get_tp_group):
        """Test scheme_reorg when stride is 1"""
        # Mock TP group with stride 1
        mock_tp_group = MagicMock()
        mock_tp_group.device_group = "mock_comm"
        mock_tp_group.world_size = 2
        mock_tp_group.rank_in_group = 0
        mock_get_tp_group.return_value = mock_tp_group

        convertor = TP_Convertor(remote_tp_size=2)
        token_num = 100
        tail_blk = 5
        kv_group = []
        local_block_ids = [1, 2, 3]
        remote_block_ids = [4, 5, 6]

        # Should return remote_block_ids unchanged when stride == 1
        result = convertor.scheme_reorg(token_num, tail_blk, kv_group,
                                        local_block_ids, remote_block_ids)
        assert result == remote_block_ids

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_scheme_reorg_with_stride(self, mock_get_tp_group):
        """Test scheme_reorg when stride > 1"""
        # Mock TP group with stride > 1
        mock_tp_group = MagicMock()
        mock_tp_group.device_group = "mock_comm"
        mock_tp_group.world_size = 8
        mock_tp_group.rank_in_group = 2
        mock_get_tp_group.return_value = mock_tp_group

        convertor = TP_Convertor(remote_tp_size=2)
        token_num = 100
        tail_blk = 5

        # Create mock KV group
        kv_group = [[torch.randn(10, 128, 64) for _ in range(2)]
                    for _ in range(4)]
        local_block_ids = [1, 2, 3, 4]
        remote_block_ids = [10, 20, 30, 40]

        # Test scheme_reorg with stride > 1
        result = convertor.scheme_reorg(token_num, tail_blk, kv_group,
                                        local_block_ids, remote_block_ids)

        # Verify result has same length as local_block_ids
        assert len(result) == len(local_block_ids)

        # Verify attributes are set correctly
        assert hasattr(convertor, 'tail_blk')
        assert hasattr(convertor, 'kv_group')
        assert hasattr(convertor, 'send_domain')
        assert hasattr(convertor, 'recv_domain')
        assert hasattr(convertor, 'recv_num')
        assert hasattr(convertor, 'send_split')
        assert hasattr(convertor, 'recv_split')

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_token_reorg_stride_1(self, mock_get_tp_group):
        """Test token_reorg when stride is 1"""
        # Mock TP group with stride 1
        mock_tp_group = MagicMock()
        mock_tp_group.device_group = "mock_comm"
        mock_tp_group.world_size = 2
        mock_tp_group.rank_in_group = 0
        mock_get_tp_group.return_value = mock_tp_group

        convertor = TP_Convertor(remote_tp_size=2)
        convertor.token_reorg()  # Should do nothing when stride == 1

    @patch('torch.distributed.all_to_all_single')
    @patch('omni_npu.connector.utils.get_tp_group')
    def test_token_reorg_with_stride(self, mock_get_tp_group,
                                     mock_all_to_all_single):
        """Test token_reorg when stride > 1"""
        # Mock TP group with stride > 1
        mock_tp_group = MagicMock()
        mock_tp_group.device_group = "mock_comm"
        mock_tp_group.world_size = 8
        mock_tp_group.rank_in_group = 2
        mock_get_tp_group.return_value = mock_tp_group

        convertor = TP_Convertor(remote_tp_size=2)

        # Set up attributes needed for token_reorg
        token_num = 100
        tail_blk = 5
        kv_group = [[torch.randn(10, 128, 64) for _ in range(2)]]
        local_block_ids = [1, 2, 3, 4]
        remote_block_ids = [10, 20, 30, 40]

        convertor.scheme_reorg(token_num, tail_blk, kv_group, local_block_ids,
                               remote_block_ids)

        # Mock recv tensor with correct dimensions
        mock_recv = MagicMock()
        mock_recv.dim.return_value = 3
        mock_recv.size.return_value = (10, 2, 64)
        mock_recv.transpose.return_value = mock_recv

        with patch('torch.Tensor.new_empty', return_value=mock_recv):
            with patch.object(TP_Convertor, 'store_kv') as mock_store_kv:
                convertor.token_reorg()

                # Verify all_to_all_single was called
                mock_all_to_all_single.assert_called_once()
                # Verify store_kv was called
                mock_store_kv.assert_called_once()

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_scheduled_list(self, mock_get_tp_group):
        """Test scheduled_list class method"""
        # Mock TP group for different ranks
        mock_tp_group1 = MagicMock()
        mock_tp_group1.rank_in_group = 0
        mock_tp_group2 = MagicMock()
        mock_tp_group2.rank_in_group = 1

        mock_get_tp_group.side_effect = [mock_tp_group1, mock_tp_group2]

        # Test that different ranks have different scheduled lists
        list_rank0 = TP_Convertor.scheduled_list()
        list_rank1 = TP_Convertor.scheduled_list()

        assert list_rank0 is not list_rank1
        assert isinstance(list_rank0, list)
        assert isinstance(list_rank1, list)

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_tail_blk_num(self, mock_get_tp_group):
        """Test tail_blk_num static method"""
        # Test various scenarios
        num = 100
        d_rank = 2
        d_size = 8
        p_size = 2
        pg = 128

        before, after = TP_Convertor.tail_blk_num(num, d_rank, d_size, p_size,
                                                  pg)
        assert isinstance(before, int)
        assert isinstance(after, int)
        assert before >= 0
        assert after >= 0

        # Test edge case with zero
        before_zero, after_zero = TP_Convertor.tail_blk_num(0, 0, 8, 2, 128)
        assert before_zero == 0
        assert after_zero == 0

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_link_to_remote(self, mock_get_tp_group):
        """Test link_to_remote static method"""
        # Test valid cases
        assert TP_Convertor.link_to_remote(0, 8, 2) == 0
        assert TP_Convertor.link_to_remote(3, 8, 2) == 0
        assert TP_Convertor.link_to_remote(4, 8, 2) == 1
        assert TP_Convertor.link_to_remote(7, 8, 2) == 1

        # Test ValueError when tp_size % remote_tp_size != 0
        with pytest.raises(ValueError):
            TP_Convertor.link_to_remote(0, 7, 2)

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_a2a_mapper(self, mock_get_tp_group):
        """Test a2a_mapper static method"""
        # Test balanced movement
        movement = [2, -1, -1]
        result = TP_Convertor.a2a_mapper(movement)
        assert result.shape == (3, 3)
        assert result.sum() == 2

        # Test complex movement
        movement = [3, -2, 1, -2]
        result = TP_Convertor.a2a_mapper(movement)
        assert result.shape == (4, 4)
        assert result.sum() == 4

        # Test zero movement
        movement = [0, 0, 0]
        result = TP_Convertor.a2a_mapper(movement)
        assert result.shape == (3, 3)
        assert result.sum() == 0

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_extract_kv(self, mock_get_tp_group):
        """Test extract_kv static method"""
        # Create mock KV tensors
        k = torch.randn(10, 128, 64)  # [blk, pg, D]
        v = torch.randn(10, 128, 64)
        kvs = [k, v]

        blk_i = 5
        domain = (10, 20)

        result = TP_Convertor.extract_kv(kvs, blk_i, domain)

        # Check shape: [T, N, D] where T=20-10=10, N=2, D=64
        assert result.shape == (10, 2, 64)

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_store_kv(self, mock_get_tp_group):
        """Test store_kv static method"""
        # Create mock destination KV tensors
        k = torch.zeros(10, 128, 64)  # [blk, pg, D]
        v = torch.zeros(10, 128, 64)
        kvs = [k, v]

        # Create source tensor
        T = 10
        N = 2
        D = 64
        x = torch.randn(T, N, D)  # [T, N, D]

        blk_i = 5
        domain = (10, 20)

        TP_Convertor.store_kv(x, kvs, blk_i, domain)

        # Check that destination tensors were updated
        assert not torch.allclose(k[blk_i][domain[0]:domain[1]],
                                  torch.zeros(T, D))
        assert not torch.allclose(v[blk_i][domain[0]:domain[1]],
                                  torch.zeros(T, D))

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_do_scheduled_kv_reorg_empty(self, mock_get_tp_group):
        """Test do_scheduled_kv_reorg with empty list"""
        # Mock TP group
        mock_tp_group = MagicMock()
        mock_tp_group.rank_in_group = 0
        mock_tp_group.all_gather.return_value = torch.tensor([0])
        mock_get_tp_group.return_value = mock_tp_group

        # Should not raise error when list is empty
        TP_Convertor.do_scheduled_kv_reorg()

    @patch('omni_npu.connector.utils.get_tp_group')
    def test_do_scheduled_kv_reorg_with_items(self, mock_get_tp_group):
        """Test do_scheduled_kv_reorg with items in list"""
        # Mock TP group
        mock_tp_group = MagicMock()
        mock_tp_group.rank_in_group = 0
        mock_tp_group.all_gather.return_value = torch.tensor([2])
        mock_get_tp_group.return_value = mock_tp_group

        # Add items to scheduled list
        scheduled_list = TP_Convertor.scheduled_list()

        # Create mock convertors with actual token_reorg method mocked
        mock_convertor1 = MagicMock(spec=TP_Convertor)
        mock_convertor1.transfer_done = True
        mock_convertor1.kv_group = [[torch.randn(1, 1, 1)]]

        mock_convertor2 = MagicMock(spec=TP_Convertor)
        mock_convertor2.transfer_done = True
        mock_convertor2.kv_group = [[torch.randn(1, 1, 1)]]

        mock_convertor3 = MagicMock(spec=TP_Convertor)
        mock_convertor3.transfer_done = False
        mock_convertor3.kv_group = [[torch.randn(1, 1, 1)]]

        scheduled_list.extend(
            [mock_convertor1, mock_convertor2, mock_convertor3])

        TP_Convertor.do_scheduled_kv_reorg()

        # Verify token_reorg was called on completed items
        mock_convertor1.token_reorg.assert_called_once()
        mock_convertor2.token_reorg.assert_called_once()
        mock_convertor3.token_reorg.assert_not_called()

    @patch("omni_npu.connector.utils.get_tp_group")
    def test_extract_kv_rejects_bad_tensor(self, mock_get_tp_group):
        with pytest.raises(TypeError, match="Expected torch.Tensor"):
            TP_Convertor.extract_kv(["not-a-tensor"], 0, (0, 1))
        bad = torch.randn(4, 8)
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            TP_Convertor.extract_kv([bad], 0, (0, 1))

    @patch("omni_npu.connector.utils.get_tp_group")
    def test_store_kv_rejects_mismatched_inputs(self, mock_get_tp_group):
        k = torch.zeros(2, 8, 4)
        v = torch.zeros(2, 8, 4)
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            TP_Convertor.store_kv(torch.zeros(2, 2), [k, v], 0, (0, 2))
        with pytest.raises(ValueError, match=r"x.size\(0\) mismatch"):
            TP_Convertor.store_kv(torch.zeros(3, 2, 4), [k, v], 0, (0, 2))
        with pytest.raises(ValueError, match=r"x.size\(1\) mismatch"):
            TP_Convertor.store_kv(torch.zeros(2, 3, 4), [k, v], 0, (0, 2))
        with pytest.raises(TypeError, match="Expected torch.Tensor"):
            TP_Convertor.store_kv(torch.zeros(2, 1, 4), ["bad"], 0, (0, 2))
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            TP_Convertor.store_kv(torch.zeros(2, 1, 4), [torch.zeros(8, 4)], 0, (0, 2))
        with pytest.raises(ValueError, match="D size mismatch"):
            TP_Convertor.store_kv(torch.zeros(2, 1, 4), [torch.zeros(2, 8, 8)], 0, (0, 2))

    @patch("omni_npu.connector.utils.get_tp_group")
    def test_a2a_mapper_unbalanced_raises(self, mock_get_tp_group):
        with pytest.raises(RuntimeError, match="a2a_mapper failed"):
            TP_Convertor.a2a_mapper([3, 0])

    @patch("omni_npu.connector.utils.get_tp_group")
    def test_tail_blk_num_requires_decode_ge_prefill(self, mock_get_tp_group):
        with pytest.raises(ValueError, match="must be >="):
            TP_Convertor.tail_blk_num(8, 0, 2, 4, 128)


def test_get_local_ip_closes_socket():
    from omni_npu.connector.utils import get_local_ip

    mock_sock = MagicMock()
    mock_sock.getsockname.return_value = ("10.1.2.3", 0)
    with patch("omni_npu.connector.utils.socket.socket", return_value=mock_sock):
        assert get_local_ip() == "10.1.2.3"
    mock_sock.connect.assert_called_once()
    mock_sock.close.assert_called_once()


def test_start_daemon_puts_feedback():
    from omni_npu.connector.utils import start_daemon

    def task(fb):
        fb.put("ready")

    fb = start_daemon(task)
    assert fb.get(timeout=2) == "ready"


def test_calm_down_default_interval_sleeps():
    from omni_npu.connector.utils import calm_down

    if hasattr(calm_down, "_ts"):
        calm_down._ts.pop("default-interval", None)
    times = iter([100.0, 100.001])
    sleeps = []
    with patch("omni_npu.connector.utils.time.time", side_effect=lambda: next(times)):
        with patch("omni_npu.connector.utils.time.sleep", side_effect=sleeps.append):
            calm_down("default-interval")
            calm_down("default-interval")
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(0.009)


class TestParallelDesc:
    def test_from_list_and_rank(self):
        from omni_npu.connector.utils import ParallelDesc

        desc = ParallelDesc([2, 2, 1, 4, 1], rank=5)
        assert desc.to_list() == [2, 2, 1, 4, 1]
        assert desc.size == 16
        assert desc.rank == 5
        assert desc.pp_rank == 0
        assert desc.dp_rank == 1
        assert desc.pcp_rank == 0
        assert desc.tp_rank == 1
        assert desc.dcp_rank == 0

    def test_invalid_inputs(self):
        from omni_npu.connector.utils import ParallelDesc

        with pytest.raises(ValueError, match="5 dims"):
            ParallelDesc([1, 2, 3])
        with pytest.raises(ValueError, match="positive int"):
            ParallelDesc([1, 2, 1, 0, 1])
        with pytest.raises(ValueError, match="invalid parallel"):
            ParallelDesc("bad")
        with pytest.raises(ValueError, match="invalid rank"):
            ParallelDesc([1, 1, 1, 1, 1], rank=2)

    def test_from_parallel_config_and_vllm_config(self):
        from vllm.config import ParallelConfig, VllmConfig
        from omni_npu.connector.utils import ParallelDesc

        pc = ParallelConfig(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=2,
            decode_context_parallel_size=1,
        )
        desc = ParallelDesc(pc, rank=1)
        assert desc.tp == 2
        assert desc.tp_rank == 1

        vllm_config = MagicMock(spec=VllmConfig)
        vllm_config.parallel_config = pc
        desc2 = ParallelDesc(vllm_config)
        assert desc2.to_list() == [1, 1, 1, 2, 1]

    def test_from_runtime_groups(self):
        from omni_npu.connector.utils import ParallelDesc

        def grp(size, rank=0):
            return MagicMock(world_size=size, rank_in_group=rank)

        with patch(MODULE + ".get_pp_group", return_value=grp(2)), patch(
            MODULE + ".get_dp_group", return_value=grp(1)
        ), patch(MODULE + ".get_pcp_group", return_value=grp(1)), patch(
            MODULE + ".get_tp_group", return_value=grp(4)
        ), patch(MODULE + ".get_dcp_group", return_value=grp(1)), patch(
            MODULE + ".get_world_group", return_value=grp(8, 3)
        ):
            desc = ParallelDesc(None)
            assert desc.size == 8
            assert desc.rank == 3
            assert desc.tp_rank == 3

        with patch(MODULE + ".get_pp_group", return_value=grp(1)), patch(
            MODULE + ".get_dp_group", return_value=grp(1)
        ), patch(MODULE + ".get_pcp_group", return_value=grp(1)), patch(
            MODULE + ".get_tp_group", return_value=grp(1)
        ), patch(MODULE + ".get_dcp_group", return_value=grp(1)), patch(
            MODULE + ".get_world_group", return_value=grp(2, 0)
        ):
            with pytest.raises(ValueError, match="world_size"):
                ParallelDesc(None)


def _wait_until(pred, timeout=2.0, interval=0.01):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


def _zmq_err(errno):
    import zmq

    err = zmq.ZMQError()
    err.errno = errno
    return err


def _pump_until(server, client, replies, key, **ops):
    def ready():
        server.handle(**ops)
        client.routine()
        return key in replies

    assert _wait_until(ready), f"missing reply key {key}"


class TestSimpleServerClient:
    def test_query_echo_and_unknown_op(self):
        from omni_npu.connector.utils import SimpleClient, SimpleServer

        server = SimpleServer("tcp://127.0.0.1:*")
        client = SimpleClient(server.addr())
        replies = {}

        def echo(req):
            return {"echo": req.get("v")}

        client.query("echo", cb=lambda r: replies.update(ok=r), v=7)
        _pump_until(server, client, replies, "ok", echo=echo)
        assert replies["ok"]["echo"] == 7

        client.query("missing", cb=lambda r: replies.update(err=r))
        _pump_until(server, client, replies, "err", echo=echo)
        assert replies["err"]["err"] == "unknown op"
        client.close()
        del server

    def test_send_json_type_error_and_timeout(self):
        from omni_npu.connector.utils import SimpleClient, SimpleServer

        server = SimpleServer("tcp://127.0.0.1:*")
        client = SimpleClient(server.addr())
        with pytest.raises(TypeError, match="req must be dict"):
            client.send_json("bad")
        timed = []
        client.send_json({"op": "noop"}, cb=lambda r: timed.append(r))
        rest = client.routine(timeout=-1.0)
        assert rest == 0
        assert timed == [None]
        client.close()
        leftover = client.send_json({"op": "after-close"})
        assert leftover == 0
        del server

    def test_bad_req_and_internal_error(self):
        from omni_npu.connector.utils import SimpleServer
        import zmq

        server = SimpleServer("tcp://127.0.0.1:*")
        ctx = zmq.Context()
        sock = ctx.socket(zmq.DEALER)
        sock.connect(server.addr())
        sock.send_string("not-json")
        assert _wait_until(lambda: self._pump_and_poll(server, sock))
        assert "bad_req" in sock.recv_string()

        sock.send_string('{"req_id": 1, "req": "not-dict"}')
        assert _wait_until(lambda: self._pump_and_poll(server, sock))
        assert "bad_req" in sock.recv_string()

        sock.send_string('{"req_id": 2, "req": {"op": "boom"}}')

        def boom(_req):
            raise RuntimeError("explode")

        raised = {"ok": False}

        def pump_boom():
            try:
                server.handle(boom=boom)
            except RuntimeError as exc:
                if "explode" in str(exc):
                    raised["ok"] = True
                    return True
                raise
            return False

        assert _wait_until(pump_boom)
        assert raised["ok"] is True
        sock.close(linger=0)
        ctx.term()
        del server

    @staticmethod
    def _pump_and_poll(server, sock):
        server.handle()
        return sock.poll(10) != 0

    def test_todo_nonblock_returns_when_empty(self):
        from omni_npu.connector.utils import SimpleServer

        server = SimpleServer("tcp://127.0.0.1:*")
        assert list(server.todo(loop=False)) == []
        del server

    def test_client_skips_malformed_reply_and_flush(self):
        from omni_npu.connector.utils import SimpleClient, SimpleServer
        import zmq

        server = SimpleServer("tcp://127.0.0.1:*")
        client = SimpleClient(server.addr())
        with patch.object(
            client.sock,
            "recv_string",
            side_effect=["not-json", _zmq_err(zmq.EAGAIN)],
        ):
            assert client.routine() == 0
        client.flush(polling=0.0, timeout=0.0)
        client.close()
        del server

    def test_todo_reraises_non_eagain(self):
        from omni_npu.connector.utils import SimpleServer
        import zmq

        server = SimpleServer("tcp://127.0.0.1:*")
        with patch.object(server.sock, "recv_multipart", side_effect=_zmq_err(zmq.ETERM)):
            with pytest.raises(zmq.ZMQError):
                list(server.todo(loop=False))
        del server

    def test_client_routine_raises_after_close_and_non_eagain(self):
        from omni_npu.connector.utils import SimpleClient, SimpleServer
        import zmq

        server = SimpleServer("tcp://127.0.0.1:*")
        client = SimpleClient(server.addr())
        client.close()
        with pytest.raises(RuntimeError, match="called after stop"):
            client.routine()
        client2 = SimpleClient(server.addr())
        with patch.object(client2.sock, "recv_string", side_effect=_zmq_err(zmq.ETERM)):
            with pytest.raises(zmq.ZMQError):
                client2.routine()
        client2.close()
        del server


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
