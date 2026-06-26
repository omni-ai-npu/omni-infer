# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""ZMQ transport layer for OmniCache connector."""

import uuid
from typing import Dict, List, Optional

import msgpack
import zmq
from vllm.logger import init_logger

logger = init_logger(__name__)


class RouterDealerClient:
    """ZMQ Router-Dealer pattern client."""

    def __init__(self, server_address="tcp://localhost:5555"):
        """Initialize the ZMQ client.

        Args:
            server_address: ZMQ server address.
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)

        client_id = f"client_{uuid.uuid4().hex[:8]}".encode('utf-8')
        self.socket.setsockopt(zmq.IDENTITY, client_id)

        self.socket.connect(server_address)
        print(f"Connected to server at {server_address} with ID: {client_id.decode()}")

    def send_request(
        self,
        request_id: str,
        cluster_id: int,
        src_id_list: List[int],
        dst_id_list: List[int],
        rank_id: int,
        src_dp_rank: int = 0
    ) -> bool:
        """Send a request to the server.

        Args:
            request_id: Unique request ID.
            cluster_id: Target cluster ID.
            src_id_list: Source block IDs.
            dst_id_list: Destination block IDs.
            rank_id: Rank ID.
            src_dp_rank: Remote DP rank (ox applies offset internally).

        Returns:
            True if send successful, False otherwise.
        """
        try:
            request_data = {
                'request_id': request_id,
                'table_id': rank_id,
                'src_block_ids': src_id_list,
                'dst_block_ids': dst_id_list,
                'cluster_id': cluster_id,
                'src_dp_rank': src_dp_rank
            }
            packed_data = msgpack.packb(request_data)
            self.socket.send(packed_data)
            return True
        except Exception as e:
            print(f"Error sending request {request_id}: {e}")
            return False

    def receive_response(self, timeout: int = 1000) -> Optional[Dict]:
        """Receive response from the server.

        Args:
            timeout: Timeout in milliseconds.

        Returns:
            Response dict or None if timeout/error.
        """
        try:
            if self.socket.poll(timeout, zmq.POLLIN):
                response_data = self.socket.recv()
                response = msgpack.unpackb(response_data)
                return response
            else:
                logger.warning("self.socket.poll(timeout, zmq.POLLIN) is False")
        except Exception as e:
            print(f"Error receiving response: {e}")
        return None

    def close(self):
        """Close the ZMQ client."""
        self.socket.close()
        self.context.term()
        print("Client closed")


class ZMQSendProxy:
    """Proxy for sending ZMQ requests through a queue."""

    def __init__(self, send_q: "multiprocessing.Queue"):
        """Initialize the proxy.

        Args:
            send_q: Multiprocessing queue for sending items.
        """
        self._q = send_q

    def send_request(
        self,
        request_id: str,
        cluster_id: int,
        src_id_list: List[int],
        dst_id_list: List[int],
        rank_id: int,
        src_dp_rank: int = 0
    ) -> bool:
        """Send a request through the queue.

        Args:
            request_id: Unique request ID.
            cluster_id: Target cluster ID.
            src_id_list: Source block IDs.
            dst_id_list: Destination block IDs.
            rank_id: Rank ID.
            src_dp_rank: Remote DP rank (ox applies offset internally).

        Returns:
            Always True (queue put doesn't fail).
        """
        from omni_cache.connector.utils.metadata import _SendItem
        self._q.put(_SendItem(
            request_id=request_id,
            cluster_id=int(cluster_id),
            src_ids=list(src_id_list),
            dst_ids=list(dst_id_list),
            rank_id=int(rank_id),
            src_dp_rank=int(src_dp_rank),
        ))
        return True
