import json
import os

GLOBAL_RANK_TABLE_KEY_ENV = 'GLOBAL_RANK_TABLE_FILE_PATH_KEY'
POD_IP_ENV = 'POD_IP'


def get_local_device_size():
    try:
        pod_ip = os.getenv(POD_IP_ENV)

        global_rank_table_path_key = os.getenv(GLOBAL_RANK_TABLE_KEY_ENV)
        global_rank_table_path = os.getenv(global_rank_table_path_key)
        with open(global_rank_table_path, 'r') as file:
            buf = file.read()
        rank_table = json.loads(buf)

        for group in rank_table['server_group_list']:
            for server in group['server_list']:
                if server['server_ip'] == pod_ip:
                    if 'device' in server:
                        return len(server['device'])
                    else:
                        return 0
        return -1
    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":
    print(get_local_device_size())
