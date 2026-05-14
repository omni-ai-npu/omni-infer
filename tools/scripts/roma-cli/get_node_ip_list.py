import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser(description="node ip help")
parser.add_argument('--mode', default='all')

args = parser.parse_args()

# Get file path
json_file_path = os.getenv('GLOBAL_RANK_TABLE_FILE_PATH')

# Check if environment variable is set
if not json_file_path:
    print("Environment variable GLOBAL_RANK_TABLE_FILE_PATH is not set")
    sys.exit(1)

# Try to open and read JSON file
while True:
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file {json_file_path}: {e}")
        sys.exit(1)
    if data.get('status', "") != "completed":
        time.sleep(30)
    else:
        break

# Find IP addresses
all_ip = ""
for group in data.get('server_group_list', []):
    for server in group['server_list']:
        all_ip = all_ip + "," + server.get('server_ip')

if all_ip.startswith(","):
    all_ip = all_ip[1:]

# Output found IP addresses
if all_ip:
    print(all_ip)
else:
    print("No IP addresses found")
    sys.exit(1)