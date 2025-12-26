import os
import argparse
import socket

parser = argparse.ArgumentParser(description="node ip help")
parser.add_argument('--mode', default='all')
args = parser.parse_args()


def get_ip_from_domain(domain):
    try:
        ip = socket.gethostbyname(domain)
        return ip
    except socket.gaierror:
        return ""

VC_WORKER_HOSTS = os.getenv('VC_WORKER_HOSTS')
vc_worker_hosts_list = VC_WORKER_HOSTS.split(',') if VC_WORKER_HOSTS else []
vc_worker_ip_list = [get_ip_from_domain(host) for host in vc_worker_hosts_list]  # 去掉端口号
num = len(vc_worker_ip_list)
all_ip = ",".join(vc_worker_ip_list)
print(all_ip)
