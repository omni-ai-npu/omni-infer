# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
import torch
import zmq

from omni_npu.connector.mm_feature_transfer.config import NetworkConnectorConfig
from omni_npu.connector.mm_feature_transfer.mm_feature_connector.network_connector import (
    NetworkMMFeatureConnector,
    BaseMMFeatureConnector,
    NetworkTransportFactory,
    pack_complex_payload,
    unpack_complex_payload,
)

# ----------  Fixtures ----------
@pytest.fixture
def sender_config():
    config = NetworkConnectorConfig()
    config.is_producer = True
    config.remote_endpoints = "tcp://127.0.0.1:5555"
    config.high_water_mark = 1000
    config.linger_ms = 0
    return config

@pytest.fixture
def connected_sender(sender_config):
    """
    返回一个已模拟连接状态的 Sender 实例。
    - 不调用真实的 connect()，避免真实 ZMQ 依赖。
    - 手动设置 _connected=True 并注入 AsyncMock 的 socket。
    """
    sender = NetworkTransportFactory.create(sender_config)
    sender._connected = True
    sender.socket = AsyncMock()  # 模拟 socket，所有方法均为 AsyncMock
    sender.context = None  # 不需要 context，因为 send 中未使用
    return sender

@pytest.fixture
def receiver_config():
    config = NetworkConnectorConfig()
    config.is_producer = False
    config.local_endpoint = "tcp://127.0.0.1:5555"
    return config

@pytest.fixture
def receiver(receiver_config):
    return NetworkTransportFactory.create(receiver_config)

@pytest.fixture
def bound_receiver(receiver_config):
    receiver = NetworkTransportFactory.create(receiver_config)
    receiver._bound = True
    receiver.socket = AsyncMock()
    receiver.poller = AsyncMock()
    return receiver

class TestZmqAsyncPubSender:
    @pytest.mark.asyncio
    async def test_send_success(self, connected_sender):
        """
        场景：正常发送成功
        预期：返回 True，且 send_multipart 被正确调用一次（参数正确，flags=zmq.NOBLOCK）
        """
        hash_key = "test_key"
        payload = b"test_payload"

        # Mock send_multipart normally return (no exception)
        connected_sender.socket.send_multipart = AsyncMock(return_value=None)

        result = await connected_sender.send(hash_key, payload)

        assert result is True
        connected_sender.socket.send_multipart.assert_awaited_once_with(
            [hash_key.encode("utf-8"), payload],
            flags=zmq.NOBLOCK,
        )

    @pytest.mark.asyncio
    async def test_send_not_connected(self, connected_sender):
        """
        场景：未连接状态（_connected=False 或 socket 为 None）
        预期：返回 False，并记录 warning 日志，不尝试发送
        """
        # 测试 _connected=False
        connected_sender._connected = False
        with patch("omni_npu.connector.mm_feature_transfer.mm_feature_connector.network_connector.logger") as mock_logger:
            result = await connected_sender.send("key", b"data")
            assert result is False
            mock_logger.warning.assert_called_once_with("send() called before connect()")

        # 测试 socket=None（即使 _connected=True）
        connected_sender._connected = True
        connected_sender.socket = None
        with patch("omni_npu.connector.mm_feature_transfer.mm_feature_connector.network_connector.logger") as mock_logger:
            result = await connected_sender.send("key", b"data")
            assert result is False
            mock_logger.warning.assert_called_once_with("send() called before connect()")

    @pytest.mark.asyncio
    async def test_send_queue_full(self, connected_sender):
        """
        场景：发送队列满，抛出 zmq.Again
        预期：返回 False，记录 debug 日志（包含 DROP 和 QUEUE_FULL），不记录 warning
        """
        hash_key = "key"
        payload = b"data"

        # Let send_multipart throw zmq.Again
        connected_sender.socket.send_multipart = AsyncMock(side_effect=zmq.Again())

        with patch("omni_npu.connector.mm_feature_transfer.mm_feature_connector.network_connector.logger") as mock_logger:
            result = await connected_sender.send(hash_key, payload)
            assert result is False
            mock_logger.info.assert_called_once_with(
                "ZmqAsyncSubSender[DROP] key=%s, size=%s, reason=QUEUE_FULL",
                hash_key,
                len(payload),
            )

    @pytest.mark.asyncio
    async def test_send_zmq_error(self, connected_sender):
        """
        场景：其他 ZMQ 错误（如网络错误），抛出 zmq.ZMQError
        预期：返回 False，记录 debug 日志（含 errno）和 warning 日志（通用信息）
        """
        hash_key = "key"
        payload = b"data"

        # Mock a not Again ZMQError
        error = zmq.ZMQError(errno=zmq.EPROTONOSUPPORT)
        connected_sender.socket.send_multipart = AsyncMock(side_effect=error)

        with patch("omni_npu.connector.mm_feature_transfer.mm_feature_connector.network_connector.logger") as mock_logger:
            result = await connected_sender.send(hash_key, payload)
            assert result is False
            mock_logger.info.assert_called_once_with(
                "ZmqAsyncSubSender[DROP] key=%s, size=%s, reason=NETWORK_ERROR_%s",
                hash_key,
                len(payload),
                error.errno,
            )
            mock_logger.warning.assert_called_once_with("ZMQ send error: %s", error)


class TestZmqAsyncSubReceiver:
    @pytest.mark.asyncio
    async def test_receive_success(self, bound_receiver):
        """
        场景：成功接收一条消息
        预期：poller.poll 返回事件列表，socket.recv_multipart 返回 multipart 数据，
            返回 (hash_key, payload)
        """
        hash_key = "test_key"
        payload = b"test_payload"
        hash_bytes = hash_key.encode("utf-8")

        bound_receiver.poller.poll = AsyncMock(return_value=[(bound_receiver.socket, zmq.POLLIN)])
        bound_receiver.socket.recv_multipart = AsyncMock(return_value=[hash_bytes, payload])

        result = await bound_receiver.receive()

        assert result == (hash_key, payload)
        bound_receiver.poller.poll.assert_awaited_once_with(timeout=bound_receiver.poll_timeout)
        bound_receiver.socket.recv_multipart.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_receive_not_bound(self, bound_receiver):
        """
        场景：未绑定状态（_bound=False 或 socket/poller 为 None）
        预期：返回 None，并记录 warning 日志
        """
        with patch("omni_npu.connector.mm_feature_transfer.mm_feature_connector.network_connector.logger") as mock_logger:
            # 情况1：_bound=False
            bound_receiver._bound = False
            result = await bound_receiver.receive()
            assert result is None
            mock_logger.warning.assert_called_once_with("receive() called before bind()")
            mock_logger.reset_mock()

            # 情况2：socket=None 但 _bound=True（模拟异常状态）
            bound_receiver._bound = True
            bound_receiver.socket = None
            bound_receiver.poller = None
            result = await bound_receiver.receive()
            assert result is None
            mock_logger.warning.assert_called_once_with("receive() called before bind()")

    @pytest.mark.asyncio
    async def test_receive_timeout(self, bound_receiver):
        """
        场景：poll 超时，无数据
        预期：poller.poll 返回空列表，receive 返回 None
        """
        # 模拟 poller.poll 返回空列表（超时）
        bound_receiver.poller.poll = AsyncMock(return_value=[])

        result = await bound_receiver.receive()

        assert result is None
        bound_receiver.poller.poll.assert_awaited_once_with(timeout=bound_receiver.poll_timeout)
        bound_receiver.socket.recv_multipart.assert_not_called()

    @pytest.mark.asyncio
    async def test_receive_zmq_error(self, bound_receiver):
        """
        场景：接收过程中抛出 zmq.ZMQError（非超时类错误）
        预期：捕获异常，记录 warning 日志，返回 None
        """
        bound_receiver.poller.poll = AsyncMock(return_value=[(bound_receiver.socket, zmq.POLLIN)])
        error = zmq.ZMQError(errno=zmq.EPROTONOSUPPORT)
        bound_receiver.socket.recv_multipart = AsyncMock(side_effect=error)

        with patch("omni_npu.connector.mm_feature_transfer.mm_feature_connector.network_connector.logger") as mock_logger:
            result = await bound_receiver.receive()
            assert result is None
            mock_logger.warning.assert_called_once_with(f"Receive error: {error}")
            bound_receiver.poller.poll.assert_awaited_once()
            bound_receiver.socket.recv_multipart.assert_awaited_once()

    def test_bind_success(self, receiver):
        """
        场景：首次绑定成功
        预期：创建 Context、Socket，设置选项，绑定地址，创建 Poller 并注册，返回 True
        """
        with patch("zmq.asyncio.Context") as MockContext:
            # 创建 mock context 和 socket
            mock_context = Mock()
            mock_socket = Mock()
            mock_context.socket.return_value = mock_socket
            MockContext.return_value = mock_context

            with patch("zmq.asyncio.Poller") as MockPoller:
                mock_poller = Mock()
                MockPoller.return_value = mock_poller

                result = receiver.bind()

                assert result is True
                assert receiver._bound is True

                mock_context.socket.assert_called_once_with(zmq.SUB)
                mock_socket.bind.assert_called_once_with(receiver.bind_endpoint)
                mock_poller.register.assert_called_once_with(mock_socket, zmq.POLLIN)

    def test_bind_address_in_use(self, receiver):
        """
        场景：端口被占用（多进程竞争），bind 抛出 zmq.ZMQError 且 errno 为 EADDRINUSE
        预期：返回 False，不重新抛出异常，但内部资源已被创建（需注意清理，但代码中未清理，测试仅验证返回 False）
        """
        with patch("zmq.asyncio.Context") as MockContext:
            mock_context = Mock()
            mock_socket = Mock()
            receiver.close = Mock()
            mock_context.socket.return_value = mock_socket
            MockContext.return_value = mock_context

            error = zmq.ZMQError(errno=zmq.EADDRINUSE)
            mock_socket.bind.side_effect = error

            result = receiver.bind()

            assert result is False
            assert receiver._bound is False
            receiver.close.assert_called_once()

# ----------  Fixtures ----------

@pytest.fixture
def mock_local_conn():
    conn = AsyncMock(spec=BaseMMFeatureConnector)
    conn.save = AsyncMock(return_value=None)
    return conn

@pytest.fixture
def mock_transport():
    transport = AsyncMock()
    transport.send = AsyncMock(return_value=True)
    transport.receive = AsyncMock(return_value=None)  
    transport.endpoints = "test_endpoint"
    transport.bind = Mock(return_value=False)
    transport.close = Mock()
    # Make sure isinstance(transport, NetworkReceiver) is False
    transport.__class__ = object
    return transport

@pytest.fixture
def connector(mock_local_conn, mock_transport):
    with patch("omni_npu.connector.mm_feature_transfer.mm_feature_connector.network_connector.NetworkTransportFactory.create", return_value=mock_transport):

        config = NetworkConnectorConfig()
        config.is_producer = True
        # 构造函数会检查 isinstance(transport, NetworkReceiver)，我们已设置为 False，不会启动线程
        conn = NetworkMMFeatureConnector(config, mock_local_conn)
        conn.transport = mock_transport
    return conn


class TestNetworkMMFeatureConnector:
    @pytest.mark.asyncio
    async def test_save_normal(self, connector, mock_local_conn, mock_transport):
        mm_hash = "hash123"
        metadata = {"key": "value"}
        tensors = {"t1": torch.randn(2, 3)}
        serialized_updates = [{"op": "add", "data": 1}]

        # Mock pack_complex_payload 返回固定字节
        with patch("omni_npu.connector.mm_feature_transfer.mm_feature_connector.network_connector.pack_complex_payload", return_value=b"packed_data") as mock_pack:
            await connector.save(mm_hash, metadata, tensors, serialized_updates)

            await asyncio.sleep(0)
            mock_local_conn.save.assert_awaited_once_with(mm_hash, metadata, tensors, serialized_updates)
            mock_pack.assert_called_once_with(
                metadata=metadata, tensors=tensors, serialized_state=serialized_updates
            )
            mock_transport.send.assert_awaited_once_with(mm_hash, b"packed_data")


    @pytest.mark.asyncio
    async def test_save_local_save_fails(self, connector, mock_local_conn):
        """本地保存失败：local_conn.save 不抛出异常，回调记录错误"""
        mm_hash = "hash123"
        metadata = {"key": "value"}
        tensors = {"t1": torch.randn(2, 3)}
        serialized_updates = [{"op": "add", "data": 1}]

        exception = RuntimeError("Disk full")
        mock_local_conn.save.side_effect = exception

        with patch("omni_npu.connector.mm_feature_transfer.mm_feature_connector.network_connector.logger") as mock_logger:
            await connector.save(mm_hash, metadata, tensors, serialized_updates)
            # 给事件循环一点时间执行回调
            await asyncio.sleep(1)
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0]
            assert call_args[0] == "MMFeatureConnector local save failed for %s: %s"
            assert call_args[1] == mm_hash
            assert "Disk full" in str(call_args[2])


    @pytest.mark.asyncio
    async def test_save_transport_send_fails(self, connector, mock_local_conn, mock_transport):
        """网络发送返回 False：save 应正常结束，仅记录 success=False 的 debug 日志"""
        mm_hash = "hash123"
        metadata = {"key": "value"}
        tensors = {"t1": torch.randn(2, 3)}
        serialized_updates = [{"op": "add", "data": 1}]

        mock_transport.send.return_value = False

        with patch("omni_npu.connector.mm_feature_transfer.mm_feature_connector.network_connector.pack_complex_payload", return_value=b"packed"):
            with patch("omni_npu.connector.mm_feature_transfer.mm_feature_connector.network_connector.logger") as mock_logger:
                await connector.save(mm_hash, metadata, tensors, serialized_updates)

                mock_transport.send.assert_awaited_once()
                # 验证 debug 日志中包含 success=False
                mock_logger.debug.assert_any_call(
                    "MMFeatureConnector sent MM feature %s to %s: %s",
                    mm_hash,
                    mock_transport.endpoints,
                    False,
                )


    @pytest.mark.asyncio
    async def test_run_forever_normal(self, connector, mock_local_conn, mock_transport):
        """normal receive: receive 返回数据，解包成功，创建保存任务，回调注册"""
        mm_hash = "hash123"
        payload = b"packed_data"
        metadata = {"key": "value"}
        tensors = {"t1": torch.randn(2, 3)}
        state = [{"op": "add"}]

        # set receive to return data in the first time, 
        # None in the second time and stop the loop (_running = False)
        receive_calls = 0

        async def receive_side_effect():
            nonlocal receive_calls
            receive_calls += 1
            if receive_calls == 1:
                return (mm_hash, payload)
            else:
                connector._running = False
                return None

        mock_transport.receive.side_effect = receive_side_effect

        with patch("omni_npu.connector.mm_feature_transfer.mm_feature_connector.network_connector.unpack_complex_payload", return_value=(metadata, tensors, state)) as mock_unpack:
            with patch("omni_npu.connector.mm_feature_transfer.mm_feature_connector.network_connector.logger") as mock_logger:
                # will exit after the second call of receive
                await connector.run_forever()

                mock_unpack.assert_called_once_with(payload)
                await asyncio.sleep(0)
                mock_local_conn.save.assert_awaited_once_with(mm_hash, metadata, tensors, state)
                mock_logger.debug.assert_any_call(
                    "MMFeatureConnector received MM feature %s", mm_hash
                )


    @pytest.mark.asyncio
    async def test_run_forever_receive_timeout(self, connector, mock_transport):
        """接收超时/空数据：receive 返回 None，不创建任何保存任务，循环继续"""
        # set receive to return None, and set _running=False in the second call to exit loop
        receive_calls = 0

        async def receive_side_effect():
            nonlocal receive_calls
            receive_calls += 1
            if receive_calls == 1:
                return None  # timeout
            else:
                connector._running = False
                return None

        mock_transport.receive.side_effect = receive_side_effect

        with patch("omni_npu.connector.mm_feature_transfer.mm_feature_connector.network_connector.logger") as mock_logger:
            with patch("asyncio.create_task") as mock_create_task:
                await connector.run_forever()
                mock_create_task.assert_not_called()


    @pytest.mark.asyncio
    async def test_run_forever_local_save_fails(self, connector, mock_local_conn, mock_transport):
        """接收后本地保存失败：回调应记录错误，但接收循环继续"""
        mm_hash = "hash123"
        payload = b"packed"
        metadata = {"key": "value"}
        tensors = {"t1": torch.randn(2, 3)}
        state = [{"op": "add"}]

        exception = RuntimeError("Save failed")
        mock_local_conn.save.side_effect = exception

        receive_calls = 0

        async def receive_side_effect():
            nonlocal receive_calls
            receive_calls += 1
            if receive_calls == 1:
                return (mm_hash, payload)
            else:
                connector._running = False
                return None

        mock_transport.receive.side_effect = receive_side_effect

        with patch("omni_npu.connector.mm_feature_transfer.mm_feature_connector.network_connector.unpack_complex_payload", return_value=(metadata, tensors, state)):
            with patch("omni_npu.connector.mm_feature_transfer.mm_feature_connector.network_connector.logger") as mock_logger:
                # local save task will fail and trigger the callback
                await connector.run_forever()
                await asyncio.sleep(1)
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args[0]
                assert call_args[0] == "MMFeatureConnector local save failed for key %s: %s"
                assert call_args[1] == mm_hash
                assert "Save failed" in str(call_args[2])


class TestPayloadPackUnpack:
    def test_float32_tensor_cpu(self):
        """测试 float32 张量的完整往返，验证数值一致性和元数据保留"""
        original_metadata = {"id": 42, "description": "test"}
        original_tensors = {
            "matrix": torch.randn(3, 4, dtype=torch.float32, device="cpu"),
            "vector": torch.arange(10, dtype=torch.float32, device="cpu")
        }
        original_state = [{"step": 0, "loss": 0.5}, {"step": 1, "loss": 0.3}]

        packed = pack_complex_payload(original_metadata, original_tensors, original_state)
        unpacked_metadata, unpacked_tensors, unpacked_state = unpack_complex_payload(packed)

        assert unpacked_metadata == original_metadata
        assert unpacked_state == original_state
        assert set(unpacked_tensors.keys()) == set(original_tensors.keys())
        for name in original_tensors:
            orig = original_tensors[name]
            recon = unpacked_tensors[name]
            assert recon.dtype == orig.dtype
            assert recon.device == torch.device("cpu")
            assert torch.allclose(orig, recon, atol=1e-6, rtol=1e-5)
            assert recon.shape == orig.shape

    def test_bfloat16_conversion(self):
        """测试 bfloat16 张量的转换，验证数值在 float32 中间表示下无损"""
        original_tensors = {
            "bf16_tensor": torch.randn(2, 5, dtype=torch.bfloat16, device="cpu")
        }
        original_metadata = {"test": "bfloat16"}
        original_state = []

        packed = pack_complex_payload(original_metadata, original_tensors, original_state)
        _, unpacked_tensors, _ = unpack_complex_payload(packed)

        recon = unpacked_tensors["bf16_tensor"]
        orig = original_tensors["bf16_tensor"]

        # 解包后的 dtype 应为 bfloat16
        assert recon.dtype == torch.bfloat16
        assert recon.device == torch.device("cpu")

        # 由于 bfloat16 转 float32 再转回 bfloat16 是精确的（因为 bfloat16 是截断 float32），
        # 转换前后转为 float32 后应完全一致
        assert torch.equal(orig.float(), recon.float())