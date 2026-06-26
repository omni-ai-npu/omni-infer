# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class RecordingHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.server.received.append(self.path)
        self._send_response()

    def do_GET(self):
        self.server.received.append(self.path)
        self._send_response()

    def _send_response(self):
        status = getattr(self.server, "status_code", 200)
        body = getattr(self.server, "body", b"{\"ok\":true}")

        # Support special response patterns for JSON escape testing

            # Cover all escape cases: " \ \n \r \t \b \f
        body = b'{"message": "quote: \" backslash: \\ newline: \n crlf: \r\n tab: \t backspace: \b formfeed: \f"}'

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def start_mock_server(port, status_code=200, body=b"{\"ok\":true}"):
    server = ThreadingHTTPServer(("127.0.0.1", port), RecordingHandler)
    server.received = []
    server.status_code = status_code
    server.body = body
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def stop_mock_server(server):
    server.shutdown()
    server.server_close()