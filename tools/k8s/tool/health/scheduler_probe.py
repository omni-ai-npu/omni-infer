import requests
import argparse
import sys
import json


def check_health(ip, port):
    url = f"http://{ip}:{port}/status"

    try:
        response = requests.get(url, timeout=5)

        # 检查HTTP状态码是否为2xx
        if not response.ok:
            print(f"ERROR: Health check failed - HTTP {response.status_code}", file=sys.stderr)
            return False

        # 解析JSON响应
        try:
            data = response.json()
            status = data.get('status')

            if status != 'running':
                print(f"ERROR: Service status is '{status}', expected 'running'", file=sys.stderr)
                return False

            print("OK: Service is healthy and running")
            return True

        except json.JSONDecodeError:
            print("ERROR: Invalid JSON response", file=sys.stderr)
            return False

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request failed - {str(e)}", file=sys.stderr)
        return False


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Service health checker')
    parser.add_argument('--ip', required=True, help='Service IP address')
    parser.add_argument('--port', required=True, help='Service port number')

    args = parser.parse_args()

    # 执行健康检查
    success = check_health(args.ip, args.port)

    # 返回适当的退出码
    sys.exit(0 if success else 1)
