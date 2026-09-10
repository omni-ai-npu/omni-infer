import json
import logging
import os
import sys
import time

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.StreamHandler(sys.stdout))
_logger.setLevel(logging.INFO)


def main():
    json_file_path = os.getenv('GLOBAL_RANK_TABLE_FILE_PATH')
    if not json_file_path:
        return 1

    while True:
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        except Exception:
            return 1
        if data.get('status', "") != "completed":
            time.sleep(30)
        else:
            break

    parts = []
    for group in data.get('server_group_list', []):
        if not isinstance(group, dict) or 'server_list' not in group:
            continue
        ips = [
            server.get('server_ip')
            for server in group['server_list']
            if isinstance(server, dict) and 'server_ip' in server
        ]
        if ips:
            parts.append(','.join(ips))
    all_ip = ';'.join(parts)

    if all_ip:
        _logger.info(all_ip)
    else:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
