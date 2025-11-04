from typing import Any
import tempfile  # 如果使用IPC方式，需要/tmp路径
import zmq
from enum import Enum


CommMethod = Enum("comm_method", ("TCP", "IPC"))
SocketType = Enum("socket_type", ("Sender", "Receiver"))

class ZmqComm:
    def __init__(self, ip, port, dp_rank, comm_method, socket_type) -> None:
        self.ip = ip
        self.port = port
        self.dp_rank = dp_rank
        self.comm_method = comm_method
        self.socket_type = socket_type

        self.POLL_INTERVAL_IN_MILLISECONDS = 200

        self.setup()
    
    def setup(self) -> None:
        self.zmq_ctx = zmq.Context()

        if self.comm_method == CommMethod.TCP:
            self.addr = f"tcp://{self.ip}:{self.port}"
        elif self.comm_method == CommMethod.IPC:
            self.addr = f"ipc://{tempfile.gettempdir()}/zmq_comm_dp_{self.dp_rank}"
        
        if self.socket_type == SocketType.Receiver:
            self.socket = self.zmq_ctx.socket(zmq.PULL)
            self.socket.bind(self.addr)
            self.poller = zmq.Poller()
            self.poller.register(self.socket, zmq.POLLIN)
        if self.socket_type == SocketType.Sender:
            self.socket = self.zmq_ctx.socket(zmq.PUSH)
            self.socket.connect(self.addr)
    
    def terminate(self) -> None:
        self.socket.close()
        self.zmq_ctx.term()
    
    def send(self, data: Any) -> None:
        self.socket.send_pyobj(data)

    def recv(self) -> Any:
        # 用Poller轮询，100ms时间范围内
        # 收到消息则返回内容，没收到消息则返回None
        # 目的是为了防止阻塞在recv_pyobj()上导致无法手动退出
        if self.poller.poll(self.POLL_INTERVAL_IN_MILLISECONDS):
            return self.socket.recv_pyobj()
        else:
            return None